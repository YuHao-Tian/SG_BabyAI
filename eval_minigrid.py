import argparse
import math
import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from tqdm import trange


def make_env(env_id: str, seed: int):
    env = gym.make(env_id, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)
    env.reset(seed=seed)
    return env


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a Bernoulli proportion."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / den
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return p, lo, hi


def eval_env(model, env_id: str, episodes: int, seed0: int = 0, deterministic: bool = True):
    env = make_env(env_id, seed0)

    rets = []
    succ = 0

    for ep in trange(episodes, desc=f"Eval {env_id}"):
        obs, info = env.reset(seed=seed0 + ep)
        done = False
        ep_ret = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(int(action))
            ep_ret += float(r)
            done = term or trunc

        rets.append(ep_ret)
        if ep_ret > 0: 
            succ += 1

    env.close()

    avg_ret = sum(rets) / len(rets) if rets else 0.0
    sr, sr_lo, sr_hi = wilson_ci(succ, episodes)

    return avg_ret, succ, episodes, sr, sr_lo, sr_hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--envs", nargs="+", default=[
        "BabyAI-OneRoomS8-v0",
        "BabyAI-OneRoomS12-v0",
        "BabyAI-OneRoomS16-v0",
        "BabyAI-OneRoomS20-v0",
    ])
    ap.add_argument("--seed0", type=int, default=0, help="Base random seed. Episode i uses seed = seed0 + i.")
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="If set, use deterministic (greedy) actions during evaluation for reproducible rollouts."
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for loading the policy (e.g., 'cpu' or 'cuda'). For MLP-based PPO policies, CPU is often sufficient for evaluation."
    )
    args = ap.parse_args()

    model = PPO.load(args.model, device=args.device)

    for env_id in args.envs:
        avg_ret, k, n, sr, lo, hi = eval_env(
            model, env_id, args.episodes, seed0=args.seed0, deterministic=args.deterministic or True
        )
        print(
            f"{env_id:24s} | avg_return={avg_ret:.3f} | "
            f"success={k}/{n} ({sr*100:.2f}%) | 95%CI=[{lo*100:.2f}%, {hi*100:.2f}%]"
        )


if __name__ == "__main__":
    main()
