import dataclasses
import functools
from sched import scheduler
from typing import Literal, TypeAlias, Self, Optional, Tuple
import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import sys
from jax import debug
from jax import lax
import numpy as np

@dataclasses.dataclass(frozen=True)
class ModelConfig:
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    action_chunk_size: int = 8
    
    # LoRA configuration
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_dropout: float = 0.1
    enable_lora: bool = True
    
    # Masking configuration
    enable_masking: bool = True
    mask_token_dim: int = 64
    preserve_early_count: int = 1  # Number of early actions to preserve (not mask)


def posemb_sincos(pos: jax.Array, embedding_dim: int, min_period: float, max_period: float) -> jax.Array:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def create_preserve_early_mask(shape: Tuple[int, ...], preserve_count: int = 2) -> jax.Array:
    """Create a mask that preserves the first few actions and masks ALL later ones.

    Args:
        shape: Shape of the input tensor [B, T, D]
        preserve_count: Number of early actions to preserve (not mask)

    Returns:
        mask: Boolean mask where True indicates masked positions, shape [B, T, 1]
    """
    batch_size, seq_len, _ = shape

    # Create mask that masks everything after preserve_count
    mask = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)

    # Mask all positions after preserve_count
    if preserve_count < seq_len:
        mask = mask.at[:, preserve_count:].set(True)

    # Broadcast to [B, T, 1]
    return mask[..., None]



PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    # Change the masking strategy here
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        idx = jnp.arange(total)
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        w = jnp.where(idx >= end, 1.0, w)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)
    

class LoRALinear(nnx.Module):
    """LoRA-enabled Linear layer that can be frozen for base model and trained for adaptation.
    LoRA only takes action_emb as input, and condition is integrated through FiLM technique."""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        kernel_init: bool = False, 
        use_bias: bool = True,
        enable_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        *,
        rngs: nnx.Rngs
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.enable_lora = enable_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Base layer (frozen during LoRA training)
        if kernel_init:
            self.base_layer = nnx.Linear(in_features, out_features, kernel_init=nnx.initializers.zeros_init(), use_bias=use_bias, rngs=rngs)
        else:
            self.base_layer = nnx.Linear(in_features, out_features, use_bias=use_bias, rngs=rngs)
        
        # LoRA layers (trained during LoRA fine-tuning) - only takes action_emb
        if enable_lora:
            self.lora_A = nnx.Linear(in_features, lora_rank, use_bias=False, rngs=rngs)
            self.lora_B = nnx.Linear(lora_rank, out_features, use_bias=False, rngs=rngs)
            self.lora_dropout = nnx.Dropout(lora_dropout, rngs=rngs)

            # Scaling factor
            self.lora_scaling = lora_alpha / lora_rank

            self.mask_proj = nnx.Linear(1, in_features, use_bias=True, rngs=rngs) # IF use this, need to adjust training code (optimizer) too to include it
    
    def __call__(self, x: jax.Array, training: bool = False, use_lora: bool | None = None, mask: jax.Array | None = None) -> jax.Array:
        # Base layer output
        base_output = self.base_layer(x)
        
        # Determine whether to use LoRA
        should_use_lora = self.enable_lora if use_lora is None else use_lora
        
        if not should_use_lora:
            return base_output
        
        x_lora = x
        if mask is not None and hasattr(self, "mask_proj"):
            # small learned bias from mask (mask-awareness) – does NOT change backbone stats
            x_lora = x_lora + self.mask_proj(mask)   # broadcast on feature dim

        delta = self.lora_A(x_lora)
        if training and self.lora_dropout.rate > 0.0:
            delta = self.lora_dropout(delta, deterministic=False)
        delta = self.lora_B(delta)                    # LoRA delta in output space

        # ----- Gate the LoRA delta by the mask -----
        if mask is None:
            gate = 1.0
        else:
            # broadcast mask to output feature size
            # mask: [B, ..., 1] -> [B, ..., out_features]
            gate = jnp.broadcast_to(mask, base_output.shape)

        y = base_output + gate * (self.lora_scaling * delta)
        return y


class LoRAMLPMixerBlock(nnx.Module):
    """MLPMixerBlock with LoRA-enabled linear layers."""
    
    def __init__(
        self, 
        token_dim: int, 
        token_hidden_dim: int, 
        channel_dim: int, 
        channel_hidden_dim: int, 
        enable_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        *,
        rngs: nnx.Rngs
    ):
        self.token_mix_in = LoRALinear(token_dim, token_hidden_dim, use_bias=False, 
                                      enable_lora=enable_lora, lora_rank=lora_rank, 
                                      lora_alpha=lora_alpha, lora_dropout=lora_dropout, rngs=rngs)
        self.token_mix_out = LoRALinear(token_hidden_dim, token_dim, use_bias=False,
                                       enable_lora=enable_lora, lora_rank=lora_rank,
                                       lora_alpha=lora_alpha, lora_dropout=lora_dropout, rngs=rngs)
        self.channel_mix_in = LoRALinear(channel_dim, channel_hidden_dim, use_bias=False,
                                        enable_lora=enable_lora, lora_rank=lora_rank,
                                        lora_alpha=lora_alpha, lora_dropout=lora_dropout, rngs=rngs)
        self.channel_mix_out = LoRALinear(channel_hidden_dim, channel_dim, use_bias=False,
                                         enable_lora=enable_lora, lora_rank=lora_rank,
                                         lora_alpha=lora_alpha, lora_dropout=lora_dropout, rngs=rngs)
        self.norm_1 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)

        self.adaln_1 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        self.adaln_2 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)


    def __call__(self, x: jax.Array, time_emb: jax.Array, training: bool = False, use_lora: bool = None, mask: Optional[jax.Array] = None) -> jax.Array:
        # Use time_emb for diffusion guidance (adaln layers)
        scale_1, shift_1, gate_1 = jnp.split(self.adaln_1(time_emb)[:, None], 3, axis=-1)
        scale_2, shift_2, gate_2 = jnp.split(self.adaln_2(time_emb)[:, None], 3, axis=-1)

        # token mix
        residual = x
        x = self.norm_1(x) * (1 + scale_1) + shift_1
        x = x.transpose(0, 2, 1)
        x = self.token_mix_in(x, training=training, use_lora=use_lora, mask=None)
        x = nnx.gelu(x)
        x = self.token_mix_out(x, training=training, use_lora=use_lora, mask=None)
        x = x.transpose(0, 2, 1)
        x = residual + gate_1 * x

        # channel mix
        residual = x
        x = self.norm_2(x) * (1 + scale_2) + shift_2
        x = self.channel_mix_in(x, training=training, use_lora=use_lora, mask=mask)
        x = nnx.gelu(x)
        x = self.channel_mix_out(x, training=training, use_lora=use_lora, mask=mask)
        x = residual + gate_2 * x

        return x

def charbonnier(x, eps=1e-3):  # smooth L1
    return jnp.sqrt(x * x + eps * eps)

def schedule_func(kind, step, warm, final, total_steps, k=5, warmup_steps=2000, hold_frac=0.1):
    def _to_float32(x):
        return jnp.asarray(x, dtype=jnp.float32)

    step = _to_float32(step)
    T = _to_float32(total_steps)

    def schedule_linear(_):
        """
        linearly decay from `warm` -> `final` over [0, total_steps].
        """
        t = step / jnp.maximum(1.0, T)
        p = warm + (final - warm) * t
        return jnp.clip(p, 0.0, 1.0)

    def schedule_cosine(_):
        """
        cosine decay from `warm` -> `final`.
        """
        t = step / jnp.maximum(1.0, T)
        w = 0.5 * (1.0 + jnp.cos(jnp.pi * t))        # 1 -> 0
        p = final + (warm - final) * w               # warm -> final
        return jnp.clip(p, 0.0, 1.0)

    def schedule_exp(_):
        """
        exponential (smooth at start, faster later).
        k controls steepness (larger = steeper).
        """
        t = step / jnp.maximum(1.0, T)
        w = jnp.exp(-k * t)                           # 1 -> ~0
        p = final + (warm - final) * w
        return jnp.clip(p, 0.0, 1.0)

    def schedule_piecewise(_):
        """
        optional warmup/hold, then linear decay to final.
        - warmup_steps: steps to ramp from 1.0 -> warm (rarely needed here)
        - hold_frac: fraction of total_steps to hold at `warm` before decaying
        """
        # warmup 1.0 -> warm
        def after_warmup(_):
            hold_steps = _to_float32(hold_frac) * T
            t = jnp.clip((step - _to_float32(warmup_steps) - hold_steps) / jnp.maximum(1.0, T - _to_float32(warmup_steps) - hold_steps), 0.0, 1.0)
            return warm + (final - warm) * t

        p = jax.lax.cond(step < _to_float32(warmup_steps),
                        lambda _: 1.0 - (1.0 - warm) * (step / jnp.maximum(1.0, _to_float32(warmup_steps))),
                        after_warmup,
                        operand=None)
        return jnp.clip(p, 0.0, 1.0)

    branches = (schedule_linear, schedule_cosine, schedule_exp, schedule_piecewise)
    # kind must be an integer scalar (can be traced); ensure dtype:
    return lax.switch(jnp.asarray(kind, jnp.int32), branches, operand=None)

class LoRAFlowPolicy(nnx.Module):
    """FlowPolicy with LoRA support for efficient fine-tuning."""
    
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
        rngs: nnx.Rngs,
    ):
        self.channel_dim = config.channel_dim
        self.action_dim = action_dim
        self.action_chunk_size = config.action_chunk_size
        self.enable_lora = config.enable_lora
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout
        
        # Masking configuration
        self.enable_masking = config.enable_masking
        self.preserve_early_count = config.preserve_early_count


        # Input projection with LoRA
        self.in_proj = LoRALinear(
            action_dim + obs_dim, 
            config.channel_dim,
            enable_lora=config.enable_lora,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            rngs=rngs
        )
        
        # MLP stack with LoRA
        self.mlp_stack = [
            LoRAMLPMixerBlock(
                config.action_chunk_size,
                config.token_hidden_dim,
                config.channel_dim,
                config.channel_hidden_dim,
                enable_lora=config.enable_lora,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]
        
        # Time MLP with LoRA
        self.time_mlp = nnx.Sequential(
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
        )
        
        # Final layers
        self.final_norm = nnx.LayerNorm(config.channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.final_adaln = nnx.Linear(
            config.channel_dim, 2 * config.channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.out_proj = LoRALinear(
            config.channel_dim, action_dim,
            enable_lora=config.enable_lora, lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, rngs=rngs
        )


    def __call__(self, obs_curr: jax.Array, x_t: jax.Array, time: jax.Array, training: bool = False, 
                use_lora: bool = True, mask: Optional[jax.Array] = None) -> jax.Array:
        assert x_t.shape == (obs_curr.shape[0], self.action_chunk_size, self.action_dim), x_t.shape
        
        time_emb = posemb_sincos(
            jnp.broadcast_to(time, obs_curr.shape[0]), self.channel_dim, min_period=4e-3, max_period=4.0
        )

        time_emb = self.time_mlp(time_emb)
        
        obs_curr = einops.repeat(obs_curr, "b e -> b c e", c=self.action_chunk_size)
        
        x = jnp.concatenate([x_t, obs_curr], axis=-1)

        x = self.in_proj(x, training=training and use_lora, use_lora=use_lora, mask=mask)
        
        # Apply MLP stack with LoRA
        for mlp in self.mlp_stack:
            x = mlp(x, time_emb, training=training and use_lora, use_lora=use_lora, mask=mask)
        
        assert x.shape == (obs_curr.shape[0], self.action_chunk_size, self.channel_dim), x.shape
        
        scale, shift = jnp.split(self.final_adaln(time_emb)[:, None], 2, axis=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x_curr = self.out_proj(x, training=training and use_lora, use_lora=use_lora, mask=mask)
        
        return x_curr

    def predict_without_lora(self, obs_curr: jax.Array, x_t: jax.Array, time: jax.Array) -> jax.Array:
        """
        Predict actions using only the fixed weights without LoRA.
        This method bypasses all LoRA layers and uses only the base model weights.
        """
        return self.__call__(obs_curr, x_t, time, training=False, use_lora=False)

    def action(self, obs_curr: jax.Array, action_to_execute: jax.Array,
                preserve_early_count: int, num_steps: int, training: bool = False, init: bool=False) -> jax.Array:
        """
        Sampling by integrating the learned velocity field v_t.
        - Input is UNMASKED.
        - Mask is passed as a condition to the model.
        - Known (unmasked) region is clamped each step to action_to_execute.
        """
        
        mask = create_preserve_early_mask(action_to_execute.shape, preserve_early_count)  # shape [B, T, 1]

        # No mask tokens: keep inputs in-distribution
        x0_obs = action_to_execute
        x_init = x0_obs 

        dt = 1.0 / num_steps

        def step(carry, _):
            x_t, time = carry
            # model predicts velocity v_t on UNMASKED x_t; mask is only a condition
            v_t = self(obs_curr, x_t, time, mask=mask, training=training)
            x_update = x_t + dt * v_t
            
            if init:
                return (x_update, time + dt), None
          
            x_t_next = mask * x_update + (1.0 - mask) * x0_obs
            return (x_t_next, time + dt), None

        (x_final, _), _ = jax.lax.scan(step, (x_init, 0.0), None, length=num_steps)
        assert x_final.shape == (obs_curr.shape[0], self.action_chunk_size, self.action_dim)
        return x_final

    def action_without_lora(self, rng: jax.Array, obs_curr: jax.Array, num_steps: int) -> jax.Array:
        """
        Generate actions using only the fixed weights without LoRA.
        This method performs the full action generation process without using LoRA layers.
        """
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry
            v_t = self.predict_without_lora(obs_curr, x_t, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs_curr.shape[0], self.action_chunk_size, self.action_dim))
        (x_final, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_final.shape == (obs_curr.shape[0], self.action_chunk_size, self.action_dim), x_final.shape
        return x_final


    def async_loss(self, rng: jax.Array, obs_curr: jax.Array, action_curr: jax.Array, batch_d: jax.Array,
                    gamma_flow: float = 0.01, gamma_keep: float = 0.01, step: int = 0, num_steps: int = 32, schedule: int=0, 
                    start: float=0.7, end: float = 0.2, warmup: int=4, hold: float=0.15) -> jax.Array:
        assert action_curr.dtype == jnp.float32
        assert action_curr.shape == (obs_curr.shape[0], self.action_chunk_size, self.action_dim)

        # --- RNGs ---
        noise_rng, time_rng, mask_rng, dropout_rng = jax.random.split(rng, 4)

        time  = jax.random.uniform(time_rng, (obs_curr.shape[0],))                   # [B]
        noise = jax.random.normal(noise_rng, shape=action_curr.shape)                # [B,T,D]

        # === UNMASKED input to the network ===
        # Interpolate predicted action with GT
        action_pred_curr = self.action_without_lora(mask_rng, obs_curr, 5)
       
        p = schedule_func(kind=schedule, step=step, warm=start, final=end, total_steps=num_steps, k=5, warmup_steps=warmup, hold_frac=hold)
        
        ## Original
        mix = jax.random.bernoulli(time_rng, p=p, shape=(obs_curr.shape[0], 1, 1)).astype(jnp.float32)
      
        action_init = mix * action_curr + (1.0 - mix) * jax.lax.stop_gradient(action_pred_curr)

        x_t_curr = (1.0 - time[:, None, None]) * noise + time[:, None, None] * action_init 
        u_t_curr = action_curr - noise                                              

        # --- Build the inpainting mask to match runtime ---
        # Mask as soft mask, mask[b, t] = 0 for t < d_b
        B, T, D = action_curr.shape  # or self.action_chunk_size, self.action_dim
        t = jnp.arange(T)[None, :]          # [1, T]
        d = batch_d[:, None]                # [B, 1] (int32)
        
        mask_bt = (t >= d).astype(jnp.float32)   # [B, T]; 1 = to inpaint
        mask = jnp.expand_dims(mask_bt, -1)              # [B, T, 1]

        pred_curr = self(obs_curr, x_t_curr, time, training=True, mask=mask)

        # --- Masked flow loss + tiny keep-known regularizer ---
        resid = (u_t_curr - pred_curr)
        num = jnp.sum(charbonnier(resid) * mask, axis=(1,2))     # [B]
        den = jnp.maximum(1.0, jnp.sum(mask, axis=(1,2)))        # [B]
        loss_inpaint = jnp.mean(num / den)

        # Consistency loss and Delta loss
        with_backbone = self(obs_curr, x_t_curr, time, training=True, mask=mask, use_lora=False)

        loss_consist = jnp.mean(((pred_curr - with_backbone) ** 2) * (1.0 - mask))

        delta_target = (u_t_curr - with_backbone) * mask
        delta_pred   = (pred_curr - with_backbone) * mask
        # per-sample norm
        den = jnp.maximum(1.0, mask.sum(axis=(1,2)))
        loss_delta = jnp.mean(jnp.sum((delta_target - delta_pred)**2, axis=(1,2)) / den)

        loss = gamma_flow * loss_inpaint + gamma_flow * loss_delta + gamma_keep * loss_consist
        return loss