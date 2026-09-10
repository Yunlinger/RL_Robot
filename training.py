"""Shared policy bundles and evaluation; model/statistics/config stay together."""
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.biped_env import BipedEnv
from robot_config import ROOT

SCHEMA_VERSION = 4


def simulator_fingerprint():
    paths = ("biped.urdf", "robot_config.py", "envs/biped_env.py", "envs/gait.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}



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
    directory = Path(directory).expanduser().resolve()
    config = json.loads((directory / "config.json").read_text())
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible model: retrain with the ten-servo environment")
    if config.get("simulator_fingerprint") != simulator_fingerprint():
        raise ValueError("Simulator changed since training. Retrain with the new physics/configuration.")
    return config


def load_bundle(directory, *, training=False, device="cpu", render_mode=None):
    directory = Path(directory).expanduser().resolve()
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
                               and info["path_efficiency"] >= 0.80
                               and min(info["touchdowns"]) >= 3)
                results.append({"return": total, "duration_s": duration, "distance_m": distance,
                                "mean_speed_m_s": avg_speed, "lateral_distance_m": info["y_distance"],
                                "max_abs_lateral_m": info["max_abs_lateral_m"],
                                "final_heading_deg": float(np.rad2deg(info["heading_error_rad"])),
                                "max_abs_heading_deg": float(np.rad2deg(info["max_abs_heading_rad"])),
                                "path_efficiency": info["path_efficiency"],
                                "fallen": info["is_fallen"], "touchdowns": info["touchdowns"],
                                "walking_success": walking})
                break
    return {"episodes": results, "mean_return": float(np.mean([r["return"] for r in results])),
            "mean_distance_m": float(np.mean([r["distance_m"] for r in results])),
            "mean_max_abs_lateral_m": float(np.mean([r["max_abs_lateral_m"] for r in results])),
            "mean_max_abs_heading_deg": float(np.mean([r["max_abs_heading_deg"] for r in results])),
            "mean_path_efficiency": float(np.mean([r["path_efficiency"] for r in results])),
            "fall_rate": float(np.mean([r["fallen"] for r in results])),
            "walking_success_rate": float(np.mean([r["walking_success"] for r in results]))}
