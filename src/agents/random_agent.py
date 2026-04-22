"""Random baseline agent."""

import numpy as np


class RandomAgent:
    def __init__(self, act_dim: int):
        self.act_dim = act_dim

    def act(self, obs: np.ndarray) -> int:
        return np.random.randint(self.act_dim)

    def store(self, reward: float):
        pass

    def update(self):
        pass
