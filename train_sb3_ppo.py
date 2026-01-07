import argparse
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO

def make_env(env_id: str, seed: int):
    env = gym.make(env_id, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env = FlattenObservation(env)
    env.reset(seed=seed)
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_env", type=str, default="BabyAI-OneRoomS8-v0")
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--save", type=str, default="ppo_oneroomS8.zip")
    args = ap.parse_args()

    env = make_env(args.train_env, seed=0)

    model = PPO(
        "MlpPolicy",        
        env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        learning_rate=2.5e-4,
        gamma=0.99,
        device="auto",   
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    env.close()
    print(f"[OK] saved: {args.save}")


if __name__ == "__main__":
    main()
