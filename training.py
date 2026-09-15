"""Shared policy bundles and evaluation; model/statistics/config stay together."""
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.biped_env import BipedEnv
from robot_config import ROOT

SCHEMA_VERSION = 4


def simulator_fingerprint():
    paths = ("biped.urdf", "robot_config.py", "envs/biped_env.py", "envs/gait.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def resolve_bundle_path(directory):
    """Resolve a model bundle or the ``best.json`` selector in a run directory."""
    path = Path(directory).expanduser().resolve()
    selector = path if path.is_file() else path / "best.json"
    if selector.is_file():
        selected = json.loads(selector.read_text()).get("bundle")
        if not isinstance(selected, str):
            raise ValueError(f"Invalid best-model selector: {selector}")
        path = selector.parent / selected
    return path



def make_vec_env(config, *, training, render_mode=None):
    kwargs = {key: config[key] for key in
              ("episode_len", "target_speed", "task", "domain_randomization", "imu_model")}
    if not training:
        kwargs["domain_randomization"] = False
    env = DummyVecEnv([lambda: Monitor(BipedEnv(**kwargs, render_mode=render_mode))])
    return VecNormalize(env, training=training, norm_obs=True, norm_reward=False, clip_obs=10.0)


def save_bundle(directory, model, env, config, *, replay=False):
    """A new directory prevents overwriting a different model's normalizer."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model.save(directory / "model.zip")
    env.save(directory / "vecnormalize.pkl")
    if replay:
        model.save_replay_buffer(directory / "replay_buffer.pkl")
    metadata = dict(config, schema_version=SCHEMA_VERSION, timesteps=model.num_timesteps,
                    simulator_fingerprint=simulator_fingerprint())
    # Write metadata last; incomplete saves cannot be loaded as valid bundles.
    (directory / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_config(directory):
    directory = resolve_bundle_path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible model: retrain with the ten-servo environment")
    if config.get("simulator_fingerprint") != simulator_fingerprint():
        raise ValueError("Simulator changed since training. Retrain with the new physics/configuration.")
    return config


def load_bundle(directory, *, training=False, device="cpu", render_mode=None):
    directory = resolve_bundle_path(directory)
    config = load_config(directory)
    base = make_vec_env(config, training=training, render_mode=render_mode)
    try:
        # Reuse the underlying env, replacing only the normalisation wrapper.
        env = VecNormalize.load(directory / "vecnormalize.pkl", base.venv)
        env.training = training
        env.norm_reward = False
        model = SAC.load(directory / "model.zip", env=env, device=device)
        if training:
            replay = directory / "replay_buffer.pkl"
            if not replay.is_file():
                raise FileNotFoundError("Resume requires replay_buffer.pkl; use the final bundle or --save-replay")
            model.load_replay_buffer(replay)
        return model, env, config
    except Exception:
        base.close()
        raise


def reset_replay_buffer(model, env, buffer_size):
    """Give a loaded policy a new replay buffer for policy-only fine-tuning."""
    model.buffer_size = int(buffer_size)
    model.replay_buffer = model.replay_buffer_class(
        model.buffer_size,
        env.observation_space,
        env.action_space,
        device=model.device,
        n_envs=env.num_envs,
        optimize_memory_usage=model.optimize_memory_usage,
        **model.replay_buffer_kwargs,
    )


def prefill_replay_with_policy(model, env, steps):
    """Collect non-random transitions before a policy-only fine-tuning run.

    SAC normally collects uniformly random actions during ``learning_starts``.
    That is counterproductive when starting from a robot policy that can already
    walk, so this routine stores stochastic actions from that policy instead.
    """
    if steps < model.batch_size:
        raise ValueError("policy warm-up steps must be at least the batch size")
    model.policy.set_training_mode(False)
    model._last_obs = env.reset()
    if model._vec_normalize_env is not None:
        model._last_original_obs = model._vec_normalize_env.get_original_obs()
    for _ in range(steps):
        action, buffer_action = model._sample_action(0, model.action_noise, env.num_envs)
        new_obs, rewards, dones, infos = env.step(action)
        model._store_transition(model.replay_buffer, buffer_action, new_obs, rewards, dones, infos)
    # ``learn(reset_num_timesteps=True)`` resets this counter and the observation.
    model.num_timesteps = 0
    model._episode_num = 0


def prefill_replay_with_reference(model, env, steps):
    """Seed a new SAC replay buffer with the zero-residual reference gait."""
    if steps < model.batch_size:
        raise ValueError("reference warm-up steps must be at least the batch size")
    model.policy.set_training_mode(False)
    model._last_obs = env.reset()
    if model._vec_normalize_env is not None:
        model._last_original_obs = model._vec_normalize_env.get_original_obs()
    action = np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32)
    for _ in range(steps):
        new_obs, rewards, dones, infos = env.step(action)
        model._store_transition(model.replay_buffer, action, new_obs, rewards, dones, infos)
    model.num_timesteps = 0
    model._episode_num = 0


def configure_sac_optimization(model, *, learning_rate, batch_size, train_freq,
                               gradient_steps, target_entropy, learning_starts):
    """Apply stable fine-tuning settings to a new or loaded SAC model."""
    model.learning_rate = float(learning_rate)
    model.lr_schedule = FloatSchedule(float(learning_rate))
    model.batch_size = int(batch_size)
    model.learning_starts = int(learning_starts)
    model.train_freq = model.train_freq.__class__(int(train_freq), model.train_freq.unit)
    model.gradient_steps = int(gradient_steps)
    model.target_entropy = float(target_entropy)
    optimizers = (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer)
    for optimizer in optimizers:
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate)


def evaluate(model, env, *, episodes=5, seed=10000, frame_callback=None):
    results = []
    for episode in range(episodes):
        env.seed(seed + episode)
        obs = env.reset()
        total = 0.0
        for _ in range(env.get_attr("episode_len")[0]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total += float(reward[0])
            if frame_callback is not None:
                frame_callback()
            if done[0]:
                info = infos[0]  # VecEnv has already reset: use terminal info, not new state.
                duration = info["elapsed_seconds"]
                distance = info["x_distance"]
                avg_speed = distance / duration
                walking = bool(not info["is_fallen"] and avg_speed >= 0.015
                                and info["max_abs_lateral_m"] <= 0.10
                                and info["max_abs_heading_rad"] <= np.deg2rad(25)
                               and info["max_abs_pitch_deg"] <= 15.0
                               and info["max_abs_roll_deg"] <= 15.0
                                and info["path_efficiency"] >= 0.80
                               and min(info["touchdowns"]) >= 3)
                results.append({"return": total, "duration_s": duration, "distance_m": distance,
                                "mean_speed_m_s": avg_speed, "lateral_distance_m": info["y_distance"],
                                "max_abs_lateral_m": info["max_abs_lateral_m"],
                                "final_heading_deg": float(np.rad2deg(info["heading_error_rad"])),
                                "max_abs_heading_deg": float(np.rad2deg(info["max_abs_heading_rad"])),
                                "max_abs_pitch_deg": info["max_abs_pitch_deg"],
                                "max_abs_roll_deg": info["max_abs_roll_deg"],
                                "path_efficiency": info["path_efficiency"],
                                "fallen": info["is_fallen"], "touchdowns": info["touchdowns"],
                                "walking_success": walking})
                break
    return {"episodes": results, "mean_return": float(np.mean([r["return"] for r in results])),
            "mean_distance_m": float(np.mean([r["distance_m"] for r in results])),
            "mean_max_abs_lateral_m": float(np.mean([r["max_abs_lateral_m"] for r in results])),
            "mean_max_abs_heading_deg": float(np.mean([r["max_abs_heading_deg"] for r in results])),
            "mean_max_abs_pitch_deg": float(np.mean([r["max_abs_pitch_deg"] for r in results])),
            "mean_max_abs_roll_deg": float(np.mean([r["max_abs_roll_deg"] for r in results])),
            "mean_path_efficiency": float(np.mean([r["path_efficiency"] for r in results])),
            "fall_rate": float(np.mean([r["fallen"] for r in results])),
            "walking_success_rate": float(np.mean([r["walking_success"] for r in results]))}
