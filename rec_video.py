import os

import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

from noise_shell import NoisyObservationShell


SIGMA = 0.0
SEED = 0
MODEL_PATH = f"results_300k/model_sigma{SIGMA}_seed{SEED}.zip"
VIDEO_DIR = "results_300k/video"
N_EPISODES = 5


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = NoisyObservationShell(env, sigma=SIGMA, seed=SEED)
    env = RecordVideo(env, video_folder=VIDEO_DIR, episode_trigger=lambda ep: True)

    model = PPO.load(MODEL_PATH, device="cpu")

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done = terminated or truncated
        print(f"Эпизод {ep}: награда {total:.1f}")

    env.close()
    print(f"Видео в: {VIDEO_DIR}/")


if __name__ == "__main__":
    main()