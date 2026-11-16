import itertools
import os

from train import train


SIGMAS = [0.0, 0.01, 0.05, 0.1]
SEEDS = [0, 1, 2]
TIMESTEPS = 50_000          # ← сократили до 50k
OUT_DIR = "results"


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    configs = list(itertools.product(SIGMAS, SEEDS))
    print(f"Запускаем {len(configs)} конфигураций последовательно…")

    for i, (sigma, seed) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] sigma={sigma}, seed={seed}")
        train(sigma=sigma, seed=seed, total_timesteps=TIMESTEPS, out_dir=OUT_DIR)

    print("Все обучения завершены.")