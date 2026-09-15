"""Record a policy rollout as an annotated MP4 for project demos."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import load_bundle


FPS = 25
WIDTH, HEIGHT = 1280, 720


def fit_frame(frame):
    """Letterbox the PyBullet frame into a 16:9 video canvas."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    scale = min(WIDTH / frame.shape[1], HEIGHT / frame.shape[0])
    size = (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x = (WIDTH - size[0]) // 2
    y = (HEIGHT - size[1]) // 2
    canvas[y:y + size[1], x:x + size[0]] = resized
    return canvas


def text(frame, value, origin, size=0.8, colour=(245, 245, 245), thickness=2):
    cv2.putText(frame, value, origin, cv2.FONT_HERSHEY_SIMPLEX, size,
                colour, thickness, cv2.LINE_AA)


def title_card(seconds, title, subtitle):
    frames = []
    for _ in range(round(seconds * FPS)):
        frame = np.full((HEIGHT, WIDTH, 3), (18, 24, 32), dtype=np.uint8)
        text(frame, title, (90, 285), 1.45, (235, 245, 255), 3)
        text(frame, subtitle, (94, 350), 0.75, (150, 205, 235), 2)
        frames.append(frame)
    return frames


def end_card(seconds, metrics):
    frames = []
    for _ in range(round(seconds * FPS)):
        frame = np.full((HEIGHT, WIDTH, 3), (18, 24, 32), dtype=np.uint8)
        text(frame, "RL Robot | low-cost biped prototype", (90, 180), 1.05, (235, 245, 255), 2)
        lines = [
            f"distance: {metrics['x_distance']:.3f} m / {metrics['elapsed_seconds']:.1f} s",
            f"lateral drift: {metrics['max_abs_lateral_m'] * 100:.1f} cm",
            f"heading error: {np.rad2deg(metrics['max_abs_heading_rad']):.1f} deg",
            f"pitch / roll peak: {metrics['max_abs_pitch_deg']:.1f} / {metrics['max_abs_roll_deg']:.1f} deg",
            "simulation evidence; real hardware still needs calibration",
        ]
        for i, line in enumerate(lines):
            text(frame, line, (95, 260 + i * 55), 0.72 if i < 4 else 0.58,
                 (205, 220, 232) if i < 4 else (150, 170, 185), 2)
        frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default="低成本十舵机二足机器人")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    import torch
    torch.set_num_threads(1)
    model, env, config = load_bundle(args.model, render_mode="rgb_array")
    raw = env.venv.envs[0].unwrapped
    env.seed(args.seed)
    obs = env.reset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        env.close()
        raise RuntimeError("OpenCV could not open the MP4 writer")

    metrics = {}
    thumbnail = None
    try:
        for frame in title_card(2.5, args.title, "SG90-class servos | BNO085 | SAC reference gait"):
            writer.write(frame)
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            info = infos[0]
            metrics = info
            if step % 2 == 0:
                frame = fit_frame(raw.render())
                overlay = frame.copy()
                cv2.rectangle(overlay, (25, 25), (455, 145), (8, 14, 22), -1)
                frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
                text(frame, "RL Robot / policy rollout", (50, 62), 0.72, (235, 245, 255), 2)
                text(frame, f"t = {info['elapsed_seconds']:.1f} s", (50, 95), 0.60)
                text(frame, f"x = {info['x_distance']:.2f} m   y = {info['y_distance']:.2f} m", (50, 122), 0.60)
                writer.write(frame)
                if thumbnail is None and step >= args.steps // 3:
                    thumbnail = frame.copy()
            if done[0]:
                break
        for frame in end_card(3.5, metrics):
            writer.write(frame)
    finally:
        writer.release()
        env.close()

    summary = {key: metrics[key] for key in (
        "elapsed_seconds", "x_distance", "y_distance", "max_abs_lateral_m",
        "max_abs_heading_rad", "max_abs_pitch_deg", "max_abs_roll_deg",
        "is_fallen", "touchdowns"
    ) if key in metrics}
    summary["model"] = str(args.model)
    summary["target_speed"] = config["target_speed"]
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if thumbnail is not None:
        cover = thumbnail.copy()
        overlay = cover.copy()
        cv2.rectangle(overlay, (0, 470), (WIDTH, HEIGHT), (8, 14, 22), -1)
        cover = cv2.addWeighted(overlay, 0.82, cover, 0.18, 0)
        text(cover, "LOW-COST BIPED", (65, 555), 1.55, (240, 248, 255), 3)
        text(cover, "10x 9g servos | BNO085 | reinforcement learning",
             (70, 620), 0.78, (145, 210, 240), 2)
        cover_path = args.output.with_name(args.output.stem + "_cover.jpg")
        cv2.imwrite(str(cover_path), cover)
        print(cover_path.resolve())
    print(args.output.resolve())
    print(summary_path.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
