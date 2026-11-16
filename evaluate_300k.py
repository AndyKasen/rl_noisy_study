import glob
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from noise_shell import NoisyObservationShell


N_EPISODES = 15
RESULTS_DIR = "results_300k"


def evaluate_model(model_path, sigma, seed, n_episodes=N_EPISODES):
    env = gym.make("LunarLander-v3")
    env = NoisyObservationShell(env, sigma=sigma, seed=seed)
    model = PPO.load(model_path, device="cpu")

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return rewards


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "model_sigma*_seed*.zip"))):
        name = os.path.basename(path).replace("model_", "").replace(".zip", "")
        sigma = float(name.split("_")[0].replace("sigma", ""))
        seed = int(name.split("_")[1].replace("seed", ""))
        rewards = evaluate_model(path, sigma, seed)
        rows.append({"sigma": sigma, "seed": seed,
                     "mean_reward": np.mean(rewards), "std_reward": np.std(rewards)})
        print(f"sigma={sigma}, seed={seed}: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "comparison.csv"), index=False)
    print("Сохранено:", os.path.join(RESULTS_DIR, "comparison.csv"))


if __name__ == "__main__":
    main()