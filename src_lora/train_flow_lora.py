import concurrent.futures
import dataclasses
import functools
import pathlib
import pickle
import json
from typing import Sequence, NamedTuple

import einops
from flax import struct
import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb
import sys
import eval_flow as _eval
import generate_data
import model_async_lora as _model
import train_expert
from dataclasses import replace

from jax.tree_util import tree_map_with_path, DictKey, SequenceKey, GetAttrKey


WANDB_PROJECT = "rtc-kinetix-lora"
LOG_DIR = pathlib.Path("logs-bc")

EMA_DECAY = 0.999

@dataclasses.dataclass(frozen=True)
class Config:
    # Path to the pre-trained model checkpoint
    pretrained_model_path: str
    
    run_path: str
    seq: str
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    )
    batch_size: int = 512
    num_epochs: int = 32
    seed: int = 0

    eval: _eval.EvalConfig = _eval.EvalConfig()

    learning_rate: float = 1e-4  # Lower learning rate for LoRA fine-tuning
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000

    dir_name: str = "lora_debug"

    setting: int = 1  # Default to LoRA-only training (setting=1)
    sim_epoch: int = 32
    gamma_flow: float = 0.01
    gamma_keep: float = 0.01
    
    # LoRA-specific settings
    lora_rank: int = _model.ModelConfig.lora_rank
    lora_alpha: float = _model.ModelConfig.lora_alpha
    lora_dropout: float = _model.ModelConfig.lora_dropout
    enable_lora: bool = _model.ModelConfig.enable_lora
    channel_dim: int = _model.ModelConfig.channel_dim
    
    # Gradient masking settings
    freeze_base_during_lora: bool = True  # Whether to freeze base weights during LoRA training

    # detailed training configurations
    schedule: int = 2
    schedule_start: float = 1.0
    schedule_end: float = 0.0

    # JSON file load
    json_file: str = "task_hyperparameters.json"


@struct.dataclass
class EpochCarry:
    rng: jax.Array
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[_model.LoRAFlowPolicy, nnx.Optimizer]]
    train_mask: nnx.State
    ema_state: nnx.State
    ema_steps: jax.Array     
  
class ExpCfg(NamedTuple):
    schedule: jnp.int32       # 0/1/2/3
    schedule_start:    jnp.float32
    schedule_end:      jnp.float32
    gamma_flow: jnp.float32
    gamma_keep: jnp.float32
    setting: jnp.int32
    warmup: jnp.int32
    hold: jnp.float32
    

def _name(k):
    if isinstance(k, DictKey):     return str(k.key)
    if isinstance(k, SequenceKey): return f"{k.idx}"
    if isinstance(k, GetAttrKey):  return str(k.name)
    return str(k)


def make_lora_mask_state(params_state, *, freeze_base=True):
    """Return a BOOL mask with the SAME treedef as params_state (nnx.State).
    Leaves are *Python bools* so optax.masked can branch on them safely."""
    def decide(path, _leaf):
        names = [_name(k) for k in path]
        is_lora  = any(n.startswith("lora_") for n in names)
        is_adaln = any(n in ("final_adaln", "final_norm") for n in names)
        is_time  = any(n == "time_mlp" for n in names)
        is_maskt = any(n == "mask_token" for n in names)
        is_maskproj = any(n == "mask_proj" for n in names)
        train = (is_lora or is_maskt or is_maskproj) and not (is_adaln or is_time)
        if not freeze_base:
            train = True
        return bool(train)  # <-- crucial: return Python bool, not jnp.bool_
    return tree_map_with_path(decide, params_state)


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    
    # car_launch
    if config.seq == "lr=3e-4":
        config = replace(config, level_paths=("worlds/l/mjc_half_cheetah.json", 
                                              "worlds/l/mjc_swimmer.json",  
                                              "worlds/l/chain_lander.json",
                                              "worlds/l/cartpole_thrust.json",))
    elif config.seq == "lr=1e-4":
        config = replace(config, level_paths=("worlds/l/catapult.json", 
                                              "worlds/l/mjc_walker.json", 
                                              "worlds/l/h17_unicycle.json",
                                              "worlds/l/car_launch.json",))
    elif config.seq == "lr=3e-5":
        config = replace(config, level_paths=("worlds/l/catcher_v3.json",
                                              "worlds/l/grasp_easy.json",))
    elif config.seq == "lr=1e-5":
        config = replace(config, level_paths=("worlds/l/hard_lunar_lander.json",
                                              "worlds/l/trampoline.json",))
    

    level_names = []
    for i in range(len(config.level_paths)):
        level_names.append(config.level_paths[i].split("/")[-1][:-5])

    # Load task-specific hyperparameters
    def load_task_hyperparameters():
        """Load task-specific hyperparameters from JSON file."""
        json_path = pathlib.Path(config.json_file)
        if json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            print("Warning: task_hyperparameters.json not found")
            return {}
    
    task_configs = load_task_hyperparameters()
    
    # Create batched configuration for each level
    def create_batched_config(level_names, task_configs, base_config):
        """Create batched configuration that can be passed to jax.vmap."""
        batched_cfg = {}
        
        # Core training parameters
        for param_name in ['schedule', 'schedule_start', 'schedule_end', 'setting', 'warmup', 'hold']:
            param_values = []
            for level_name in level_names:
                if level_name in task_configs and param_name in task_configs[level_name]:
                    param_values.append(task_configs[level_name][param_name])
                else:
                    # Use default value from base config
                    if param_name == "schedule":
                        param_values.append(base_config.schedule)
                    elif param_name == 'schedule_start':
                        param_values.append(base_config.schedule_start)
                    elif param_name == 'schedule_end':
                        param_values.append(base_config.schedule_end)
                    elif param_name == 'setting':
                        param_values.append(base_config.setting)
                    elif param_name == "warmup":
                        param_values.append(base_config.warmup)
                    elif param_name == "hold":
                        param_values.append(base_config.hold)
            
            # Convert to JAX array for vmap compatibility
            batched_cfg[param_name] = jnp.array(param_values)
        
        # Additional task-specific parameters (optional)
        for param_name in ['gamma_flow', 'gamma_keep']:
            param_values = []
            for level_name in level_names:
                if level_name in task_configs and param_name in task_configs[level_name]:
                    param_values.append(task_configs[level_name][param_name])
                else:
                    # Use default value from base config
                    if param_name == 'gamma_flow':
                        param_values.append(base_config.gamma_flow)
                    elif param_name == 'gamma_keep':
                        param_values.append(base_config.gamma_keep)
            
            # Convert to JAX array for vmap compatibility
            batched_cfg[param_name] = jnp.array(param_values)
        
        return batched_cfg
    
    exp_cfg = create_batched_config(level_names, task_configs, config)
    print("Batched configuration created:")
    print("Tasks:", level_names)
    for param, values in exp_cfg.items():
        print(f"{param}: {values}")
    
    batched_cfg = ExpCfg(
        schedule=jnp.asarray(exp_cfg["schedule"], jnp.int32),
        schedule_start=jnp.asarray(exp_cfg["schedule_start"], jnp.float32),
        schedule_end=jnp.asarray(exp_cfg["schedule_end"], jnp.float32),
        gamma_flow = jnp.asarray(exp_cfg["gamma_flow"], jnp.float32),
        gamma_keep = jnp.asarray(exp_cfg["gamma_keep"], jnp.float32),
        setting = jnp.asarray(exp_cfg["setting"], jnp.int32),
        warmup = jnp.asarray(exp_cfg["warmup"], jnp.int32),
        hold = jnp.asarray(exp_cfg["hold"], jnp.float32),
    )


    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    mesh = jax.make_mesh((jax.local_device_count(),), ("level",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level"))

    action_chunk_size = config.eval.model.action_chunk_size

    # load data
    def load_data(level_path: str):
        level_name = level_path.replace("/", "_").replace(".json", "")
        return dict(np.load(pathlib.Path(config.run_path) / "data" / f"{level_name}.npz"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(load_data, config.level_paths))

    with jax.default_device(jax.devices("cpu")[0]):
        # data has shape: (num_levels, num_steps, num_envs, ...)
        # flatten envs and steps together for learning
        data = jax.tree.map(lambda *x: einops.rearrange(jnp.stack(x), "l s e ... -> l (e s) ..."), *data)

        # truncate to multiple of batch size
        valid_steps = data["obs"].shape[1] - action_chunk_size + 1
        data = jax.tree.map(
            lambda x: x[:, : (valid_steps // config.batch_size) * config.batch_size + action_chunk_size - 1], data
        )
        # put on device
        data = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                x.shape,
                sharding,
                [
                    jax.device_put(y, d)
                    for y, d in zip(jnp.split(x, jax.local_device_count()), jax.local_devices(), strict=True)
                ],
            ),
            data,
        )

    data: generate_data.Data = generate_data.Data(**data)
    print(f"Truncated data to {data.obs.shape[1]:_} steps ({valid_steps // config.batch_size:_} batches)")

    obs_dim = data.obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    # Create LoRA model config
    lora_model_config = _model.ModelConfig(
        channel_dim=config.eval.model.channel_dim,
        channel_hidden_dim=config.eval.model.channel_hidden_dim,
        token_hidden_dim=config.eval.model.token_hidden_dim,
        num_layers=config.eval.model.num_layers,
        action_chunk_size=config.eval.model.action_chunk_size,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        enable_lora=config.enable_lora,
    )


    @functools.partial(jax.jit, in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def init(rng, pretrained_state):
        rng, key = jax.random.split(rng)
        # Create LoRA model with same architecture
        policy = _model.LoRAFlowPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=lora_model_config,
            rngs=nnx.Rngs(rng),
        )

        # Copy pre-trained weights to base layers
        def copy_pretrained_weights(lora_state_dict, pretrained_state_dict):
            for key, value in lora_state_dict.items():
                if key in pretrained_state_dict:
                    if isinstance(value, dict):
                        copy_pretrained_weights(value, pretrained_state_dict[key])
                    else:
                        lora_state_dict[key] = pretrained_state_dict[key]

        def nest_base_layer(d, path=()):
            if isinstance(d, dict):
                # If both bias and kernel are present, and we're not inside time_mlp or final_adaln
                if set(['bias', 'kernel']).issubset(d.keys()) and not any(
                    name in ('time_mlp', 'final_adaln', 'adaln_1', 'adaln_2') for name in path
                ):
                    return {'base_layer': d}
                else:
                    # Recurse deeper, adding current key to the path
                    return {k: nest_base_layer(v, path + (k,)) for k, v in d.items()}
            else:
                return d

        pretrained_state = nest_base_layer(pretrained_state)
        lora_state = nnx.state(policy)
        lora_state_dict = lora_state.to_pure_dict()
    
        copy_pretrained_weights(lora_state_dict, pretrained_state)

        nnx.update(policy, lora_state_dict)

        total_params = sum(x.size for x in jax.tree.leaves(nnx.state(policy, nnx.Param)))
        print(f"Total params: {total_params:,}")

        optimizer = nnx.Optimizer(
            policy,
            optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adamw(
                    optax.warmup_constant_schedule(0, config.learning_rate, config.lr_warmup_steps),
                    weight_decay=config.weight_decay,
                ),
            ),
        )
       
        # Get params as a pure PyTree for building an aligned mask
        policy_state = nnx.state(policy, nnx.Param)
        train_mask_state = make_lora_mask_state(policy_state, freeze_base=True)
        
        assert jax.tree_util.tree_structure(policy_state) == jax.tree_util.tree_structure(train_mask_state)

        labels = jax.tree_util.tree_map(lambda b: "train" if b else "freeze", train_mask_state)
        tx = optax.multi_transform(
            {
                "train":  optax.chain(optax.clip_by_global_norm(config.grad_norm_clip),
                                    optax.adamw(config.learning_rate, weight_decay=config.weight_decay)),
                "freeze": optax.set_to_zero(),  # no momentum, no WD, no updates
            },
            labels
        )
        optimizer = nnx.Optimizer(policy, tx)

        def _clone(x):  # cheap “copy” for JAX arrays
            return jnp.array(x, copy=True) if isinstance(x, jax.Array) else x
        ema_state = jax.tree_util.tree_map(_clone, policy_state)

        graphdef, train_state = nnx.split((policy, optimizer))

        # Log which parameters are frozen vs trainable
        if config.freeze_base_during_lora:
            total_params = 0
            trainable_params = 0
            frozen_params = 0
            
            def count_params(param, mask):
                nonlocal total_params, trainable_params, frozen_params
                param_size = param.size if hasattr(param, 'size') else 1
                total_params += param_size
                if mask:
                    trainable_params += param_size
                else:
                    frozen_params += param_size
                return mask
            
            jax.tree_util.tree_map(count_params, policy_state, train_mask_state)
            print(f"Parameter counts - Total: {total_params:,}, Trainable: {trainable_params:,}, Frozen: {frozen_params:,}")
            print(f"Training efficiency: {trainable_params/total_params*100:.1f}% of parameters are trainable")
            
        return EpochCarry(rng, train_state, graphdef, train_mask_state, ema_state, ema_steps=jnp.array(0, dtype=jnp.int32))

    wandb.init(project=WANDB_PROJECT)
    rng = jax.random.key(config.seed)
    
    state_dict_paths = [
        pathlib.Path(config.pretrained_model_path) / f"{p.replace('/', '_').replace('.json', '')}.pkl"
        for p in config.level_paths
    ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pretrained_states = list(executor.map(lambda path: pickle.load(open(path, "rb")), state_dict_paths))
    pretrained_states = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *pretrained_states)
    epoch_carry = init(jax.random.split(rng, len(config.level_paths)), pretrained_states)
    num_levels, _ = data.done.shape

    def ema_update(ema_state, params_state, mask_state, decay):
        def f(e, p, m):
            if e is None or p is None:
                return e
            upd = decay * e + (1.0 - decay) * p
            # If m is a Python bool -> normal branch; otherwise use jnp.where
            if isinstance(m, bool):
                return upd if m else e
            else:
                # m may be a scalar or broadcastable array
                return jnp.where(m, upd, e)
        return jax.tree_util.tree_map(f, ema_state, params_state, mask_state)

    @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def train_epoch(epoch_carry: EpochCarry, level: kenv_state.EnvState, data: generate_data.Data, 
                    permutation: jax.Array, step: int, cfg: ExpCfg):
        def train_minibatch(carry: tuple[jax.Array, nnx.State], batch_idxs: jax.Array):
            rng, train_state, ema_state, ema_steps = carry
            policy, optimizer = nnx.merge(epoch_carry.graphdef, train_state)

            rng, key = jax.random.split(rng)

            def loss_fn(policy: _model.LoRAFlowPolicy):
                obs_curr = data.obs[batch_idxs[:, 1]]
                action_chunks_curr = data.action[batch_idxs[:, 1][:, None] + jnp.arange(action_chunk_size)[None, :]]
                done_chunks_curr = data.done[batch_idxs[:, 1][:, None] + jnp.arange(action_chunk_size)[None, :]]
                done_idxs_curr = jnp.where(
                    jnp.any(done_chunks_curr, axis=-1),
                    jnp.argmax(done_chunks_curr, axis=-1),
                    action_chunk_size,
                )
                action_chunks_curr = jnp.where(
                    jnp.arange(action_chunk_size)[None, :, None] >= done_idxs_curr[:, None, None],
                    0.0,
                    action_chunks_curr,
                )
             
                return policy.async_loss(key, obs_curr, action_chunks_curr, batch_idxs[:, 2], gamma_flow=cfg.gamma_flow,
                                         gamma_keep=cfg.gamma_keep, step=step, num_steps=config.num_epochs, schedule=cfg.schedule, 
                                         start=cfg.schedule_start, end=cfg.schedule_end, warmup=cfg.warmup, hold=cfg.hold)

            loss, grads = nnx.value_and_grad(loss_fn)(policy)

            optimizer.update(grads)

            # --- EMA update over trainable (LoRA) leaves only ---
            params_state = nnx.state(policy, nnx.Param)
            ema_state    = ema_update(ema_state, params_state, epoch_carry.train_mask, EMA_DECAY)
            ema_steps    = ema_steps + jnp.array(1, dtype=jnp.int32)
            
            base_grad_norm = optax.global_norm(grads)
            info = {"loss": loss, "grad_norm": base_grad_norm}
            _, train_state = nnx.split((policy, optimizer))

            return (rng, train_state, ema_state, ema_steps), info
        # train
        (rng, train_state, ema_state, ema_steps), train_info = jax.lax.scan(
            train_minibatch, (epoch_carry.rng, epoch_carry.train_state, epoch_carry.ema_state, epoch_carry.ema_steps), permutation
        )
        train_info = jax.tree.map(lambda x: x.mean(), train_info)
        # eval
        rng, key = jax.random.split(rng)
        eval_info = {}

        video = None
        return EpochCarry(rng, train_state, epoch_carry.graphdef, epoch_carry.train_mask, ema_state, ema_steps), ({**train_info, **eval_info}, video)
    
    base = jax.random.PRNGKey(config.seed)

    def level_epoch_keys(level_id: int, epoch_idx: int, n: int):
        k = jax.random.fold_in(base, int(level_id))
        k = jax.random.fold_in(k, int(epoch_idx))
        return jax.random.split(k, n)
    
    for epoch_idx in tqdm.tqdm(range(config.num_epochs)):
        valid_pairs = []
        T = data.done.shape[1]

        for level in range(num_levels):
            # Get task-specific setting for this level
            level_setting = int(batched_cfg.setting[level])
            
            valid_start_curr = jnp.arange(0, T)
            pairs = []
            num_valid = valid_start_curr.shape[0]
            keys = level_epoch_keys(level, epoch_idx, num_valid * 2 + 1)
            subkey1s = keys[1:num_valid+1]
            subkey2s = keys[num_valid+1:]

            if level_setting == 0:
                ds = jax.vmap(lambda key: jax.random.randint(key, shape=(), minval=0, maxval=5))(subkey2s)
                es = jax.vmap(lambda key, d: jax.random.randint(key, shape=(), minval=jnp.maximum(1, d), maxval=action_chunk_size-d+1))(subkey1s, ds)
            elif level_setting == 1:
                minval = int( 4 * (1 - epoch_idx / config.num_epochs))
                ds = jax.vmap(lambda key: jax.random.randint(key, shape=(), minval=minval, maxval=5))(subkey2s)
                es = jax.vmap(lambda key, d: jax.random.randint(key, shape=(), minval=jnp.maximum(1, d), maxval=action_chunk_size-d+1))(subkey1s, ds)

            idx_prevs = valid_start_curr - es
            selected = jnp.stack([idx_prevs, valid_start_curr, ds], axis=1)
            pairs = selected
            
            assert len(pairs) > 0, f"No valid pairs found for level {level} at epoch {epoch_idx}"
            max_pairs = (pairs.shape[0] // config.batch_size) * config.batch_size

            # Shuffle pairs
            rng, key = jax.random.split(rng)
            perm = jax.random.permutation(key, pairs.shape[0])
            shuffled_pairs = pairs[perm][:max_pairs]

            # reshape pairs into batches
            num_batches = max_pairs // config.batch_size
            pairs_batched = shuffled_pairs.reshape(num_batches, config.batch_size, selected.shape[1])
            valid_pairs.append(pairs_batched)


        min_batches = min(p.shape[0] for p in valid_pairs)
        permutation = jnp.stack([p[:min_batches] for p in valid_pairs], axis=0)

        with jax.default_device(jax.devices("cpu")[0]):
            permutation = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                x.shape,
                sharding,
                [
                    jax.device_put(y, d)
                    for y, d in zip(jnp.split(x, jax.local_device_count()), jax.local_devices(), strict=True)
                ],
            ),
            permutation,
        )

        steps = jnp.array([epoch_idx] * num_levels)
        epoch_carry, (info, video) = train_epoch(epoch_carry, levels, data, permutation, steps, batched_cfg)

        for i in range(len(config.level_paths)):
            level_name = config.level_paths[i].replace("/", "_").replace(".json", "")
            wandb.log({f"{level_name}/{k}": v[i] for k, v in info.items()}, step=epoch_idx)

            log_dir = LOG_DIR / config.dir_name / str(epoch_idx)

            if video is not None:
                video_dir = log_dir / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(video_dir / f"{level_name}.mp4", video[i], fps=15)

            policy_dir = log_dir / "policies"
            policy_dir.mkdir(parents=True, exist_ok=True)
            level_train_state = jax.tree.map(lambda x: x[i], epoch_carry.train_state)
            level_ema_state = jax.tree.map(lambda x: x[i], epoch_carry.ema_state)
            with (policy_dir / f"{level_name}.pkl").open("wb") as f:
                policy, _ = nnx.merge(epoch_carry.graphdef, level_train_state)
                live_params = nnx.state(policy, nnx.Param)             # nnx.State
                ema_params = level_ema_state
                steps_i = jax.device_get(epoch_carry.ema_steps[i])
                steps_i = int(steps_i.item() if steps_i.ndim == 0 else steps_i)
                state_dict = {
                    "params":    live_params.to_pure_dict(),
                    "ema_params": ema_params.to_pure_dict(),
                    "ema_steps": steps_i,
                }
                pickle.dump(state_dict, f)
    

if __name__ == "__main__":
    tyro.cli(main)
