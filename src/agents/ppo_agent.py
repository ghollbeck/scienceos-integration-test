"""Minimal PPO agent for cooperative navigation."""

import numpy as np


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 3e-4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self._rewards = []
        # Weight matrix (simplified — real impl would use torch.nn)
        self._W = np.random.randn(act_dim, obs_dim) * 0.01

    def act(self, obs: np.ndarray) -> np.ndarray:
        logits = self._W @ obs
        probs = self._softmax(logits)
        action = np.random.choice(self.act_dim, p=probs)
        return action

    def store(self, reward: float):
        self._rewards.append(reward)

    def update(self):
        # Simplified policy gradient update
        if not self._rewards:
            return
        G = sum(r * (0.99 ** i) for i, r in enumerate(reversed(self._rewards)))
        self._W += self.lr * G * np.random.randn(*self._W.shape)
        self._rewards.clear()

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
