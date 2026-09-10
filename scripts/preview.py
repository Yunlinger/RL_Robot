"""Preview the kinematic baseline or saved SAC policy, optionally export a GIF."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.biped_env import BipedEnv
from training import load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Omit for the zero-residual reference gait')
    parser.add_argument('--gif', type=Path, help='Optional: pip install pillow')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--randomize', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    mode = 'rgb_array' if args.gif else 'human'
    frames = []
    if args.model:
        import torch
        torch.set_num_threads(1)
        model, env, _ = load_bundle(args.model, render_mode=mode)
        env.set_attr('domain_randomization', args.randomize)
        env.seed(args.seed)
        obs = env.reset()
        raw = env.venv.envs[0].unwrapped
    else:
        model = None
        env = raw = BipedEnv(render_mode=mode, domain_randomization=args.randomize, seed=args.seed)
    try:
        for step in range(args.steps):
            # Capture before stepping: a VecEnv automatically resets on the terminal step.
            if args.gif and step % 5 == 0:
                from PIL import Image
                frames.append(Image.fromarray(raw.render()))
            if model:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)
                finished, info = bool(done[0]), infos[0]
            else:
                _, _, terminated, truncated, info = env.step(np.zeros(10))
                finished = terminated or truncated
            if finished:
                break
        print({key: info[key] for key in ('elapsed_seconds', 'x_distance', 'y_distance', 'is_fallen', 'touchdowns')})
        if args.gif:
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
            print(args.gif.resolve())
    finally:
        env.close()


if __name__ == '__main__':
    main()
