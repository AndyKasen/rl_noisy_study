import os
from train import train


CONFIGS = [
    (0.0, 0),
    (0.0, 1),
    (0.1, 0),
]
TIMESTEPS = 300_000
OUT_DIR = "results_300k"


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, (sigma, seed) in enumerate(CONFIGS, 1):
        print(f"[{i}/{len(CONFIGS)}] sigma={sigma}, seed={seed}")
        train(sigma=sigma, seed=seed, total_timesteps=TIMESTEPS, out_dir=OUT_DIR)
    print("Все обучения 300k завершены.")