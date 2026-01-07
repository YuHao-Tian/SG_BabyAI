import argparse
import os
import gymnasium as gym
import minigrid  # noqa: F401 (register envs)
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed


def make_one_env(env_id: str, seed: int):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ImgObsWrapper(env)
        env = FlattenObservation(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_envs", type=str, nargs="+",
                    default=["BabyAI-OneRoomS8-v0", "BabyAI-OneRoomS12-v0"],
                    help="List of env IDs to mix during training.")
    ap.add_argument("--n_envs", type=int, default=8,
                    help="Number of parallel envs (DummyVecEnv).")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--save", type=str, default="ppo_mixscale.zip")
    ap.add_argument("--seed", type=int, default=0)

    # PPO hyperparams (keep close to your working setup)
    ap.add_argument("--n_steps", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--ent_coef", type=float, default=0.0)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--vf_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)

    args = ap.parse_args()

    set_random_seed(args.seed)
    
    env_fns = []
    for i in range(args.n_envs):
        env_id = args.train_envs[i % len(args.train_envs)]
        env_fns.append(make_one_env(env_id, seed=args.seed + i))

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        device="cpu",
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
    )

    model.learn(total_timesteps=args.timesteps)

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    model.save(args.save)
    vec_env.close()
    print(f"[OK] saved: {args.save}")
    print(f"[OK] train_envs={args.train_envs}, n_envs={args.n_envs}, timesteps={args.timesteps}")


if __name__ == "__main__":
    main()
