import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ParameterSwitchEnv(gym.Env):
    """A minimal RL env where actions select one of several predefined parameter sets.

    This is only a placeholder. The recommended real design is:
    - observation: regime, volatility, trend strength, recent pnl, drawdown
    - action: choose parameter bucket / disable strategy / reduce risk
    - reward: risk-adjusted pnl after costs
    """

    metadata = {"render_modes": []}

    def __init__(self, n_actions: int = 4, obs_dim: int = 6):
        super().__init__()
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.step_count = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(*self.observation_space.shape).astype(np.float32)
        reward = float(np.random.randn() * 0.1)
        terminated = self.step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}
