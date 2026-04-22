"""Cooperative navigation gridworld environment."""

import numpy as np
from typing import List, Tuple


class CoopNavEnv:
    def __init__(self, num_agents: int = 2, grid_size: int = 10, max_steps: int = 200):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obs_dim = 8   # pos(2) + vel(2) + goal_delta(2) + nearest_agent(2)
        self.act_dim = 5   # stay, up, down, left, right
        self._step = 0
        self._positions = np.zeros((num_agents, 2))
        self._goals = np.zeros((num_agents, 2))

    def reset(self) -> List[np.ndarray]:
        self._step = 0
        self._positions = np.random.randint(0, self.grid_size, (self.num_agents, 2)).astype(float)
        self._goals = np.random.randint(0, self.grid_size, (self.num_agents, 2)).astype(float)
        return self._get_obs()

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool, dict]:
        deltas = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]
        for i, a in enumerate(actions):
            dx, dy = deltas[a]
            self._positions[i] += [dx, dy]
            self._positions[i] = np.clip(self._positions[i], 0, self.grid_size - 1)

        distances = np.linalg.norm(self._positions - self._goals, axis=1)
        rewards = [-d / self.grid_size for d in distances]

        self._step += 1
        done = self._step >= self.max_steps or all(d < 1.0 for d in distances)
        return self._get_obs(), rewards, done, {}

    def _get_obs(self) -> List[np.ndarray]:
        obs_list = []
        for i in range(self.num_agents):
            pos = self._positions[i] / self.grid_size
            goal_delta = (self._goals[i] - self._positions[i]) / self.grid_size
            nearest = np.zeros(2)
            if self.num_agents > 1:
                others = [j for j in range(self.num_agents) if j != i]
                dists = [np.linalg.norm(self._positions[j] - self._positions[i]) for j in others]
                nearest_j = others[np.argmin(dists)]
                nearest = (self._positions[nearest_j] - self._positions[i]) / self.grid_size
            obs_list.append(np.concatenate([pos, np.zeros(2), goal_delta, nearest]))
        return obs_list
