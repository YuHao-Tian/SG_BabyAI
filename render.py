import argparse
import numpy as np
import gymnasium as gym
import minigrid 
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    raise ImportError("Missing dependency: pillow. Install: pip install pillow") from e

import imageio.v2 as imageio


def save_gif(out_path: str, frames, fps: int):
    if not out_path.lower().endswith(".gif"):
        raise ValueError(f"--out must end with .gif, got: {out_path}")
    duration = 1.0 / max(1, fps)
    with imageio.get_writer(out_path, mode="I", format="GIF", duration=duration) as w:
        for fr in frames:
            w.append_data(fr)


def overlay_text(frame: np.ndarray, text: str):
    """Draw text on top-left corner."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    pad = 6
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--env_id", type=str, required=True)
    ap.add_argument("--out", type=str, default="rollout.gif")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--tile_size", type=int, default=32, help="Global render tile size. e.g., 32/48/64")
    ap.add_argument("--resize", type=int, default=0, help="Final output size (square). 0 = no resize")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--freeze", type=int, default=24, help="Freeze N frames after done (for showing SUCCESS/FAIL)")
    ap.add_argument("--label", action="store_true", help="Overlay SUCCESS/FAIL text on final frozen frames")
    ap.add_argument("--find_success", action="store_true",
                    help="Try multiple seeds until success then save that episode.")
    ap.add_argument("--tries", type=int, default=20, help="Used with --find_success")

    args = ap.parse_args()

    # base_env: for GLOBAL rendering only
    base_env = gym.make(args.env_id, render_mode="rgb_array")
    # obs_env: wrappers for model input (must match training)
    obs_env = ImgObsWrapper(base_env)
    obs_env = FlattenObservation(obs_env)

    model = PPO.load(args.model, device="cpu")

    def render_global():
        frame = base_env.unwrapped.get_frame(
            tile_size=args.tile_size, highlight=True, agent_pov=False
        )
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        if args.resize and args.resize > 0:
            frame = np.array(
                Image.fromarray(frame).resize((args.resize, args.resize), resample=Image.NEAREST)
            )
        return frame.astype(np.uint8)

    def run_one(seed: int):
        obs, info = obs_env.reset(seed=seed)
        frames = []
        total_r = 0.0

        frames.append(render_global())

        term = trunc = False
        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, term, trunc, info = obs_env.step(int(action))
            total_r += float(r)

            frames.append(render_global())

            if term or trunc:
                break

        return frames, total_r, term, trunc

    used_seed = args.seed
    frames, total_r, term, trunc = run_one(used_seed)

    if args.find_success:
        for k in range(args.tries):
            if total_r > 0:
                break
            used_seed = args.seed + 1 + k
            frames, total_r, term, trunc = run_one(used_seed)

    # freeze final frames + optional label
    status = "SUCCESS" if total_r > 0 else "FAIL"
    if args.freeze and args.freeze > 0:
        last = frames[-1]
        for _ in range(args.freeze):
            fr = last
            if args.label:
                fr = overlay_text(fr, status)
            frames.append(fr)

    obs_env.close()  # closes base_env too

    save_gif(args.out, frames, fps=args.fps)

    print(f"[OK] saved {args.out} | env={args.env_id} | seed={used_seed} | "
          f"frames={len(frames)} | total_r={total_r:.3f} | term={term} trunc={trunc} | {status}")


if __name__ == "__main__":
    main()
