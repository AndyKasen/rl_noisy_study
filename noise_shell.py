import gymnasium as gym
import numpy as np


class NoisyObservationShell(gym.ObservationWrapper):
    
    # Обёртка: добавка шума к наблюдениям.
    # - sigma = 0 -> без шума (базовое условие)
    # - sigma > 0 -> шум с std = sigma


    def __init__(self, env, sigma=0.0, seed=None):
        super().__init__(env)          
        self.sigma = sigma             # сила шума (std)
        self.rng = np.random.default_rng(seed)  # генератор случайных чисел

    def observation(self, obs):
        if self.sigma == 0.0:
            return obs                 # без шума
        noise = self.rng.normal(0.0, self.sigma, size=obs.shape)  # шум
        return obs + noise             # добавляем шум к наблюдению

    #Проверка

if __name__ == "__main__":

    env = gym.make("LunarLander-v3")
    env = NoisyObservationShell(env, sigma=0.0, seed=42)
    obs, _ = env.reset(seed=0)
    print("без шума:", obs[:3])

    env2 = gym.make("LunarLander-v3")
    env2 = NoisyObservationShell(env2, sigma=0.1, seed=42)
    obs2, _ = env2.reset(seed=0)
    print("с шумом:", obs2[:3])
    print("разница:", obs2[:3] - obs[:3])