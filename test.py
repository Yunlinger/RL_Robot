"""Evaluate a policy bundle; GUI is optional and import has no side effects."""
import argparse
import json
from pathlib import Path

import torch

from training import evaluate, load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Bundle directory, e.g. runs/my_run/final")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-walking", action="store_true", help="Exit 1 unless every episode passes the walking heuristic")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    torch.set_num_threads(1)
    model, env, _ = load_bundle(args.model, render_mode="human" if args.gui else None)
    try:
        metrics = evaluate(model, env, episodes=args.episodes, seed=args.seed)
        print(json.dumps(metrics, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        if args.require_walking and metrics["walking_success_rate"] < 1:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
