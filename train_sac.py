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
from training import (configure_sac_optimization, evaluate, load_bundle, make_vec_env,
                      prefill_replay_with_policy, prefill_replay_with_reference,
                      reset_replay_buffer, save_bundle)


def choose_device(requested):
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS 不可用。请使用 --device cpu，或安装支持当前 macOS/芯片的 PyTorch。")
    # SAC MLPs are small; SB3 auto does not choose MPS. Use predictable CPU by default.
    return "cpu" if requested == "auto" else requested


class EvaluationCallback(BaseCallback):
    def __init__(self, config, run_dir, frequency, episodes, save_replay,
                 early_stop_rate, early_stop_patience):
        super().__init__()
        self.config, self.run_dir = config, run_dir
        self.frequency, self.episodes = frequency, episodes
        self.save_replay = save_replay
        self.eval_env = make_vec_env(config, training=False)
        self.best_quality = None
        self.early_stop_rate = early_stop_rate
        self.early_stop_patience = early_stop_patience
        self.consecutive_target_evaluations = 0

    def _on_step(self):
        if self.n_calls % self.frequency:
            return True
        sync_envs_normalization(self.training_env, self.eval_env)
        metrics = evaluate(self.model, self.eval_env, episodes=self.episodes)
        metrics["timesteps"] = self.num_timesteps
        with (self.run_dir / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        for key in ("mean_return", "mean_distance_m", "mean_max_abs_lateral_m",
                    "mean_max_abs_heading_deg", "mean_max_abs_pitch_deg", "mean_max_abs_roll_deg",
                    "mean_path_efficiency", "fall_rate",
                    "walking_success_rate"):
            self.logger.record("eval/" + key, metrics[key])
        name = f"step_{self.num_timesteps:09d}"
        save_bundle(self.run_dir / name, self.model, self.training_env, self.config, replay=self.save_replay)
        quality = (metrics["walking_success_rate"], -metrics["fall_rate"],
                   metrics["mean_path_efficiency"], -metrics["mean_max_abs_lateral_m"],
                   -metrics["mean_max_abs_heading_deg"],
                   -metrics["mean_max_abs_pitch_deg"], -metrics["mean_max_abs_roll_deg"],
                   metrics["mean_return"])
        if self.best_quality is None or quality > self.best_quality:
            self.best_quality = quality
            (self.run_dir / "best.json").write_text(json.dumps({"bundle": name, **metrics}, indent=2) + "\n")
        print(f"Evaluation {self.num_timesteps}: distance={metrics['mean_distance_m']:.3f} m, "
              f"max_lateral={metrics['mean_max_abs_lateral_m']:.3f} m, "
              f"max_heading={metrics['mean_max_abs_heading_deg']:.1f} deg, "
              f"fall={metrics['fall_rate']:.0%}, walking={metrics['walking_success_rate']:.0%}", flush=True)
        if metrics["walking_success_rate"] >= self.early_stop_rate:
            self.consecutive_target_evaluations += 1
        else:
            self.consecutive_target_evaluations = 0
        if self.early_stop_rate > 0 and self.consecutive_target_evaluations >= self.early_stop_patience:
            print(f"Early stop: walking rate stayed at least {self.early_stop_rate:.0%} for "
                  f"{self.early_stop_patience} evaluations. Best selector: {self.run_dir / 'best.json'}", flush=True)
            return False
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
    parser.add_argument("--target-speed", type=float, default=0.04,
                        help="Desired forward speed in m/s; use the proven reference-gait default")
    parser.add_argument("--episode-len", type=int, default=600)
    parser.add_argument("--randomize", action="store_true", help="Vary mass/friction/servo strength/latency/IMU noise")
    parser.add_argument("--imu", choices=("bno085", "mpu6050"), default="bno085")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Bundle directory with replay buffer; preserves saved task/config")
    parser.add_argument("--init-model", type=Path,
                        help="Start a new fine-tuning run from a checkpoint or best.json; replay is rebuilt")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--learning-starts", type=int, default=0,
                        help="Start learning immediately from reference-gait transitions")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Small default prevents a stable gait from drifting during fine-tuning")
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-freq", type=int, default=8,
                        help="Collect this many steps for each gradient update")
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-entropy", type=float, default=-8.0,
                        help="Low exploration entropy preserves the reference gait")
    parser.add_argument("--policy-warmup-steps", type=int, default=10_000,
                        help="Policy-generated transitions before --init-model learns")
    parser.add_argument("--reference-warmup-steps", type=int, default=10_000,
                        help="Zero-residual reference transitions before a new policy learns")
    parser.add_argument("--early-stop-rate", type=float, default=0.8,
                        help="Stop after sustained walking success; set 0 to disable")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--save-replay", action="store_true", help="Also save replay in intermediate checkpoints")
    parser.add_argument("--tensorboard", action="store_true", help="Optional: pip install tensorboard")
    args = parser.parse_args()
    for name in ("steps", "threads", "episode_len", "eval_freq", "eval_episodes", "batch_size",
                 "buffer_size", "train_freq", "gradient_steps", "early_stop_patience"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_starts < 0:
        parser.error("--learning-starts must be non-negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.policy_warmup_steps < args.batch_size:
        parser.error("--policy-warmup-steps must be at least --batch-size")
    if args.reference_warmup_steps < args.batch_size:
        parser.error("--reference-warmup-steps must be at least --batch-size")
    if not 0 <= args.early_stop_rate <= 1:
        parser.error("--early-stop-rate must be between 0 and 1")
    if args.resume and args.init_model:
        parser.error("Use either --resume or --init-model, not both")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    run_dir = (args.run_dir or ROOT / "runs" / datetime.now().strftime("sac_%Y%m%d_%H%M%S_%f")).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {"episode_len": args.episode_len, "target_speed": args.target_speed, "task": args.task,
              "domain_randomization": args.randomize, "imu_model": args.imu, "seed": args.seed}
    if args.resume:
        model, env, config = load_bundle(args.resume, training=True, device=device)
        print("Resuming saved task and parameters:", config)
    elif args.init_model:
        model, env, source_config = load_bundle(args.init_model, training=False, device=device)
        env.training = True
        config = {key: source_config[key] for key in
                  ("episode_len", "target_speed", "task", "domain_randomization", "imu_model", "seed")}
        config["initial_policy"] = str(args.init_model.expanduser())
        reset_replay_buffer(model, env, args.buffer_size)
        prefill_replay_with_policy(model, env, args.policy_warmup_steps)
        print(f"Fine-tuning policy from {args.init_model}; rebuilt replay with "
              f"{args.policy_warmup_steps} policy transitions.")
    else:
        with BipedEnv(episode_len=args.episode_len, target_speed=args.target_speed, task=args.task) as checked:
            check_env(checked, warn=True)
        env = make_vec_env(config, training=True)
        model = SAC("MlpPolicy", env, device=device, verbose=0, seed=args.seed,
                    policy_kwargs={"net_arch": [128, 128]}, learning_starts=args.learning_starts,
                    batch_size=args.batch_size, buffer_size=args.buffer_size,
                    learning_rate=args.learning_rate, ent_coef="auto_0.0001",
                    target_entropy=args.target_entropy, train_freq=args.train_freq,
                    gradient_steps=args.gradient_steps, gamma=0.99, tau=0.005,
                    tensorboard_log=str(run_dir / "tensorboard") if args.tensorboard else None)
        # Start close to the feasible reference gait; residual exploration is learned.
        torch.nn.init.zeros_(model.actor.mu.weight)
        torch.nn.init.zeros_(model.actor.mu.bias)
        torch.nn.init.zeros_(model.actor.log_std.weight)
        torch.nn.init.constant_(model.actor.log_std.bias, -3.0)
        prefill_replay_with_reference(model, env, args.reference_warmup_steps)
        print(f"Seeded replay with {args.reference_warmup_steps} zero-residual reference transitions.", flush=True)
    configure_sac_optimization(
        model, learning_rate=args.learning_rate, batch_size=args.batch_size,
        train_freq=args.train_freq, gradient_steps=args.gradient_steps,
        target_entropy=args.target_entropy,
        learning_starts=0 if args.init_model else args.learning_starts,
    )
    config["optimization"] = {
        "learning_rate": args.learning_rate, "buffer_size": model.buffer_size,
        "batch_size": args.batch_size, "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps, "target_entropy": args.target_entropy,
    }
    callback = None
    try:
        callback = EvaluationCallback(config, run_dir, args.eval_freq, args.eval_episodes, args.save_replay,
                                      args.early_stop_rate, args.early_stop_patience)
        print(f"Training on {device}; output: {run_dir}", flush=True)
        interrupted = False
        try:
            model.learn(total_timesteps=args.steps, callback=callback,
                        reset_num_timesteps=not bool(args.resume), progress_bar=False)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; saving model, normalizer and replay buffer.")
        save_bundle(run_dir / "final", model, env, config, replay=True)
        if (run_dir / "best.json").exists():
            print(f"Best evaluated policy: {run_dir / 'best.json'}", flush=True)
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
