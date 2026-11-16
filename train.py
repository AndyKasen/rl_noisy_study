import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from noise_shell import NoisyObservationShell


class RewardCollector(BaseCallback):
    """Собирает награды по эпизодам во время обучения."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # info[0] — словарь текущего эпизода
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:                       # эпизод закончился
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


def makeEnv(sigma, seed):
    #Создаёт среду + обёртку шума + Monitor
    env = gym.make("LunarLander-v3")
    env = NoisyObservationShell(env, sigma=sigma, seed=seed)
    env = Monitor(env)          # Monitor собирает награды автоматически
    env.reset(seed=seed)
    return env


def train(sigma, seed, total_timesteps, out_dir):
    #Обучает PPO для одной пары (sigma, seed)
    os.makedirs(out_dir, exist_ok=True)

    env = makeEnv(sigma, seed)
    model = PPO(
        "MlpPolicy", env,
        seed=seed,
        verbose=0,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        device="cpu"
    )

    logger = RewardCollector()
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=logger)
    elapsed = time.time() - t0

    # сохранение
    tag = f"sigma{sigma}_seed{seed}"
    model.save(os.path.join(out_dir, f"model_{tag}.zip"))
    with open(os.path.join(out_dir, f"rewards_{tag}.json"), "w") as f:
        json.dump({
            "sigma": sigma,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "elapsed_sec": elapsed,
            "episode_rewards": logger.episode_rewards,
            "episode_lengths": logger.episode_lengths,
        }, f)

    print(f"[{tag}] обучено за {elapsed:.1f}с, эпизодов: {len(logger.episode_rewards)}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--out", type=str, default="results")
    args = p.parse_args()

    train(args.sigma, args.seed, args.timesteps, args.out)