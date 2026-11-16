import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(values, window=10):
    #Скользящее среднее для сглаживания кривой.
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_learning_curves_50k():
    #График 1: кривые обучения для 12 конфигураций 50k.
    fig, ax = plt.subplots(figsize=(10, 5))
    for path in sorted(glob.glob("results/rewards_*.json")):
        with open(path) as f:
            data = json.load(f)
        rewards = data["episode_rewards"]
        label = f"σ={data['sigma']}, seed={data['seed']}"
        ax.plot(smooth(rewards), alpha=0.6, label=label)
    ax.set_title("Кривые обучения (50k шагов, 12 конфигураций)")
    ax.set_xlabel("Эпизод")
    ax.set_ylabel("Награда за эпизод (сглажено, окно=10)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plot_learning_50k.png", dpi=120)
    plt.close()
    print("Сохранено: results/plot_learning_50k.png")


def plot_300k_comparison():
    #График 2: сравнение 300k (sigma=0 vs sigma=0.1).
    df = pd.read_csv("results_300k/comparison.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    for sigma, group in df.groupby("sigma"):
        ax.bar(str(sigma), group["mean_reward"].mean(),
               yerr=group["mean_reward"].std(), capsize=8,
               label=f"σ={sigma}")
    ax.set_title("Сравнение 300k: влияние шума (среднее по seed ± std)")
    ax.set_xlabel("σ (std шума)")
    ax.set_ylabel("Средняя награда (15 эпизодов)")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results_300k/plot_300k_comparison.png", dpi=120)
    plt.close()
    print("Сохранено: results_300k/plot_300k_comparison.png")


def plot_budget_comparison():
    #График 3: 50k vs 300k (влияние бюджета).
    df_50 = pd.read_csv("results/comparison.csv")
    df_300 = pd.read_csv("results_300k/comparison.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    budgets = ["50k", "300k"]
    sigma0 = [df_50[df_50["sigma"] == 0.0]["mean_reward"].mean(),
              df_300[df_300["sigma"] == 0.0]["mean_reward"].mean()]
    sigma01 = [df_50[df_50["sigma"] == 0.1]["mean_reward"].mean(),
               df_300[df_300["sigma"] == 0.1]["mean_reward"].mean()]

    x = np.arange(len(budgets))
    ax.bar(x - 0.2, sigma0, 0.4, label="σ=0.0")
    ax.bar(x + 0.2, sigma01, 0.4, label="σ=0.1")
    ax.set_xticks(x)
    ax.set_xticklabels(budgets)
    ax.set_title("Влияние бюджета обучения: 50k vs 300k")
    ax.set_xlabel("Бюджет (шагов)")
    ax.set_ylabel("Средняя награда")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/plot_budget_comparison.png", dpi=120)
    plt.close()
    print("Сохранено: results/plot_budget_comparison.png")


if __name__ == "__main__":
    plot_learning_curves_50k()
    plot_300k_comparison()
    plot_budget_comparison()
    print("Все графики готовы.")