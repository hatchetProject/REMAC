import concurrent.futures
import dataclasses
import functools
import pathlib
import pickle
from typing import Sequence

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

import eval_flow as _eval
import generate_data
# import model as _model
import model as _model
import train_expert
from dataclasses import replace

WANDB_PROJECT = "rtc-kinetix-bc"
LOG_DIR = pathlib.Path("logs-bc")


@dataclasses.dataclass(frozen=True)
class Config:
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

    learning_rate: float = 3e-4
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000

    dir_name: str="debug"

    setting: int=0
    drop: float=0.2
    sim_epoch: int=32
    gamma: float=1.0
    aug_data: int=0
    
    # Trajectory extension settings
    extend_trajectories: bool = True
    trajectory_prefix_length: int = 8
    use_first_obs_as_prefix: bool = True


@struct.dataclass
class EpochCarry:
    rng: jax.Array
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[_model.FlowPolicy, nnx.Optimizer]]


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    # config = replace(config, level_paths=config.level_paths[:1])
    if config.seq == "top":
        config = replace(config, level_paths=config.level_paths[:4])
    elif config.seq == "mid":
        config = replace(config, level_paths=config.level_paths[4:8])
    elif config.seq == "bot":
        config = replace(config, level_paths=config.level_paths[8:12])
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

    def extend_trajectories_with_prefix(data_dict, prefix_length=8):
        """Fast extension of each trajectory with a prefix of length `prefix_length` using numpy for speed."""
        # Convert all arrays to numpy for fast slicing/concat, then back to jax at the end
        np_data = {k: np.array(v) for k, v in data_dict.items()}
        num_levels = np_data["obs"].shape[0]
        extended_data = {}

        for level in range(num_levels):
            # Get data for this level
            level_data = {k: v[level] for k, v in np_data.items()}
            done = level_data["done"]
            # Find trajectory ends and starts
            trajectory_ends = np.where(done)[0]
            trajectory_starts = np.concatenate([[0], trajectory_ends[:-1] + 1])
            num_traj = len(trajectory_starts)
            # Precompute trajectory slices
            slices = [slice(start, end + 1) for start, end in zip(trajectory_starts, trajectory_ends)]
            # For each key, build extended arrays
            for key, value in level_data.items():
                if key == "obs":
                    # Use first obs as prefix
                    obs_chunks = []
                    for s in slices:
                        traj = value[s]
                        prefix = np.repeat(traj[0][None, :], prefix_length, axis=0)
                        obs_chunks.append(np.concatenate([prefix, traj], axis=0))
                    ext = np.concatenate(obs_chunks, axis=0)
                elif key == "action":
                    # Zero prefix for actions
                    action_chunks = []
                    for s in slices:
                        traj = value[s]
                        prefix = np.zeros((prefix_length, value.shape[-1]), dtype=value.dtype)
                        action_chunks.append(np.concatenate([prefix, traj], axis=0))
                    ext = np.concatenate(action_chunks, axis=0)
                elif key in ["done", "solved", "return_", "length"]:
                    # Zero prefix for bool/scalar
                    val_chunks = []
                    for s in slices:
                        traj = value[s]
                        prefix = np.zeros(prefix_length, dtype=value.dtype)
                        val_chunks.append(np.concatenate([prefix, traj], axis=0))
                    ext = np.concatenate(val_chunks, axis=0)
                else:
                    # Zero prefix for other fields
                    val_chunks = []
                    for s in slices:
                        traj = value[s]
                        prefix_shape = (prefix_length,) + value.shape[1:]
                        prefix = np.zeros(prefix_shape, dtype=value.dtype)
                        val_chunks.append(np.concatenate([prefix, traj], axis=0))
                    ext = np.concatenate(val_chunks, axis=0)
                if key not in extended_data:
                    extended_data[key] = [ext]
                else:
                    extended_data[key].append(ext)
        # Stack all levels and convert back to jax arrays
        # Find the minimum length across all levels for each key
        for k in extended_data:
            # Convert all arrays to jnp and find min length
            arrs = [jnp.asarray(x) for x in extended_data[k]]
            min_len = min(a.shape[0] for a in arrs)
            # Clip all arrays to min_len
            arrs = [a[:min_len] for a in arrs]
            extended_data[k] = jnp.stack(arrs, axis=0)
        return extended_data

    with jax.default_device(jax.devices("cpu")[0]):
        # data has shape: (num_levels, num_steps, num_envs, ...)
        # flatten envs and steps together for learning
        data = jax.tree.map(lambda *x: einops.rearrange(jnp.stack(x), "l s e ... -> l (e s) ..."), *data)

        if config.aug_data > 0:
            data = extend_trajectories_with_prefix(
                data, 
                prefix_length=config.trajectory_prefix_length, 
            )

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

    """
    data.obs.shape = (4, 1015303, 679)
    data.action.shape = (4, 1015303, 6)
    data.done.shape = (4, 1015303)
    data.solved.shape = (4, 1015303)
    data.return_.shape = (4, 1015303)
    data.length.shape = (4, 1015303)
    """

    data: generate_data.Data = generate_data.Data(**data)
    print(f"Truncated data to {data.obs.shape[1]:_} steps ({valid_steps // config.batch_size:_} batches)")

    obs_dim = data.obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    @functools.partial(jax.jit, in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def init(rng: jax.Array) -> EpochCarry:
        rng, key = jax.random.split(rng)
        policy = _model.FlowPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.eval.model,
            rngs=nnx.Rngs(key),
        )
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
        graphdef, train_state = nnx.split((policy, optimizer))
        return EpochCarry(rng, train_state, graphdef)

    @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def train_epoch(epoch_carry: EpochCarry, level: kenv_state.EnvState, data: generate_data.Data, 
                    permutation: jax.Array, step: int, setting: int, drop: float, gamma: float):
        def train_minibatch(carry: tuple[jax.Array, nnx.State], batch_idxs: jax.Array):
            rng, train_state = carry
            policy, optimizer = nnx.merge(epoch_carry.graphdef, train_state)

            rng, key = jax.random.split(rng)

            def loss_fn(policy: _model.FlowPolicy):
                obs = data.obs[batch_idxs]
                action_chunks = data.action[batch_idxs[:, None] + jnp.arange(action_chunk_size)[None, :]]
                # zero actions after done
                done_chunks = data.done[batch_idxs[:, None] + jnp.arange(action_chunk_size)[None, :]]
                done_idxs = jnp.where(
                    jnp.any(done_chunks, axis=-1),
                    jnp.argmax(done_chunks, axis=-1),
                    action_chunk_size,
                )
                action_chunks = jnp.where(
                    jnp.arange(action_chunk_size)[None, :, None] >= done_idxs[:, None, None],
                    0.0,
                    action_chunks,
                )
                return policy.loss(key, obs, action_chunks)

            loss, grads = nnx.value_and_grad(loss_fn)(policy)
            info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
            optimizer.update(grads)
            _, train_state = nnx.split((policy, optimizer))
            return (rng, train_state), info
        
        rng, key = jax.random.split(epoch_carry.rng)
        permutation = jax.random.permutation(key, data.obs.shape[0] - action_chunk_size + 1)
        # batch
        permutation = permutation.reshape(-1, config.batch_size)
        # permutation = einops.repeat(permutation, "b n -> b n e", e=4)

        # train
        (rng, train_state), train_info = jax.lax.scan(
            train_minibatch, (epoch_carry.rng, epoch_carry.train_state), permutation
        )
        train_info = jax.tree.map(lambda x: x.mean(), train_info)
        # eval
        rng, key = jax.random.split(rng)
        eval_policy, _ = nnx.merge(epoch_carry.graphdef, train_state)
        eval_info = {}

        video = None
        return EpochCarry(rng, train_state, epoch_carry.graphdef), ({**train_info, **eval_info}, video)

    wandb.init(project=WANDB_PROJECT)
    rng = jax.random.key(config.seed)
    epoch_carry = init(jax.random.split(rng, len(config.level_paths)))
    num_levels, _ = data.done.shape

    for epoch_idx in tqdm.tqdm(range(config.num_epochs)):
        valid_pairs = []
        T = data.done.shape[1]

        for level in range(num_levels):
            mask = jnp.logical_and(
                data.done[level, action_chunk_size:T] == False,
                data.done[level, 0:T - action_chunk_size] == False
            )
            valid_start_curr = jnp.arange(action_chunk_size, T)[mask]
            # valid_start_curr = jnp.arange(action_chunk_size, T-action_chunk_size + 1)

            pairs = []
            num_valid = valid_start_curr.shape[0]
            rngs = jax.random.split(rng, num_valid * 2 + 1)
            rng = rngs[0]
            subkey1s = rngs[1:num_valid+1]
            subkey2s = rngs[num_valid+1:]

            ds = jax.vmap(lambda key: jax.random.randint(key, shape=(), minval=0, maxval=5))(subkey2s)
            es = jax.vmap(lambda key, d: jax.random.randint(key, shape=(), minval=jnp.maximum(1, d), maxval=action_chunk_size-d+1))(subkey1s, ds)

            if epoch_idx >= config.sim_epoch:
                config = replace(config, setting=1)
            idx_prevs = valid_start_curr - es

            # Only keep those where idx_prev >= 0
            mask = idx_prevs >= 0
            selected = jnp.stack([idx_prevs, valid_start_curr, es, ds], axis=1)
            pairs = selected[mask]
            
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
        permutation = jnp.stack([p[:min_batches] for p in valid_pairs], axis=0) # permutation.shape: (num_levels, min_batches, batch_size, 4)

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
        settings = jnp.array([config.setting] * num_levels)
        drops = jnp.array([config.drop] * num_levels)
        gammas = jnp.array([config.gamma] * num_levels)
        epoch_carry, (info, video) = train_epoch(epoch_carry, levels, data, permutation, steps, settings, drops, gammas)

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
            with (policy_dir / f"{level_name}.pkl").open("wb") as f:
                policy, _ = nnx.merge(epoch_carry.graphdef, level_train_state)
                state_dict = nnx.state(policy).to_pure_dict()
                pickle.dump(state_dict, f)


if __name__ == "__main__":
    tyro.cli(main)
