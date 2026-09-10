"""Train/resume SAC on Mac CPU or explicitly selected Apple MPS."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import sync_envs_normalization

from envs.biped_env import BipedEnv
from robot_config import ROOT
from training import evaluate, load_bundle, make_vec_env, save_bundle


def choose_device(requested):
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS 不可用。请使用 --device cpu，或安装支持当前 macOS/芯片的 PyTorch。")
    # SAC MLPs are small; SB3 auto does not choose MPS. Use predictable CPU by default.
    return "cpu" if requested == "auto" else requested


class EvaluationCallback(BaseCallback):
    def __init__(self, config, run_dir, frequency, episodes, save_replay):
        super().__init__()
        self.config, self.run_dir = config, run_dir
        self.frequency, self.episodes = frequency, episodes
        self.save_replay = save_replay
        self.eval_env = make_vec_env(config, training=False)
        self.best_return = float("-inf")

    def _on_step(self):
        if self.n_calls % self.frequency:
            return True
        sync_envs_normalization(self.training_env, self.eval_env)
        metrics = evaluate(self.model, self.eval_env, episodes=self.episodes)
        metrics["timesteps"] = self.num_timesteps
        with (self.run_dir / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        for key in ("mean_return", "mean_distance_m", "fall_rate", "walking_success_rate"):
            self.logger.record("eval/" + key, metrics[key])
        name = f"step_{self.num_timesteps:09d}"
        save_bundle(self.run_dir / name, self.model, self.training_env, self.config, replay=self.save_replay)
        if metrics["mean_return"] > self.best_return:
            self.best_return = metrics["mean_return"]
            (self.run_dir / "best.json").write_text(json.dumps({"bundle": name, **metrics}, indent=2) + "\n")
        print(f"Evaluation {self.num_timesteps}: distance={metrics['mean_distance_m']:.3f} m, "
              f"fall={metrics['fall_rate']:.0%}, walking={metrics['walking_success_rate']:.0%}", flush=True)
        return True

    def close(self):
        self.eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400_000, help="Additional environment steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--task", choices=("stand", "walk"), default="walk")
    parser.add_argument("--target-speed", type=float, default=0.04)
    parser.add_argument("--episode-len", type=int, default=600)
    parser.add_argument("--randomize", action="store_true", help="Vary mass/friction/servo strength/latency/IMU noise")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Bundle directory with replay buffer; preserves saved task/config")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--save-replay", action="store_true", help="Also save replay in intermediate checkpoints")
    parser.add_argument("--tensorboard", action="store_true", help="Optional: pip install tensorboard")
    args = parser.parse_args()
    for name in ("steps", "threads", "episode_len", "eval_freq", "eval_episodes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_starts < 0:
        parser.error("--learning-starts must be non-negative")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    run_dir = (args.run_dir or ROOT / "runs" / datetime.now().strftime("sac_%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {"episode_len": args.episode_len, "target_speed": args.target_speed, "task": args.task,
              "domain_randomization": args.randomize, "seed": args.seed}
    if args.resume:
        model, env, config = load_bundle(args.resume, training=True, device=device)
        print("Resuming saved task and parameters:", config)
    else:
        with BipedEnv(episode_len=args.episode_len, target_speed=args.target_speed, task=args.task) as checked:
            check_env(checked, warn=True)
        env = make_vec_env(config, training=True)
        model = SAC("MlpPolicy", env, device=device, verbose=0, seed=args.seed,
                    policy_kwargs={"net_arch": [128, 128]}, learning_starts=args.learning_starts,
                    batch_size=256, buffer_size=200_000, learning_rate=3e-4,
                    ent_coef="auto_0.01", gamma=0.99, tau=0.005,
                    tensorboard_log=str(run_dir / "tensorboard") if args.tensorboard else None)
        # Start close to the feasible reference gait; residual exploration is learned.
        torch.nn.init.zeros_(model.actor.mu.weight)
        torch.nn.init.zeros_(model.actor.mu.bias)
        torch.nn.init.zeros_(model.actor.log_std.weight)
        torch.nn.init.constant_(model.actor.log_std.bias, -2.0)
    callback = None
    try:
        callback = EvaluationCallback(config, run_dir, args.eval_freq, args.eval_episodes, args.save_replay)
        print(f"Training on {device}; output: {run_dir}", flush=True)
        interrupted = False
        try:
            model.learn(total_timesteps=args.steps, callback=callback,
                        reset_num_timesteps=not bool(args.resume), progress_bar=False)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; saving model, normalizer and replay buffer.")
        save_bundle(run_dir / "final", model, env, config, replay=True)
        if not interrupted:
            sync_envs_normalization(env, callback.eval_env)
            metrics = evaluate(model, callback.eval_env, episodes=args.eval_episodes)
            (run_dir / "final" / "evaluation.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics, indent=2), flush=True)
    finally:
        if callback:
            callback.close()
        env.close()


if __name__ == "__main__":
    main()
