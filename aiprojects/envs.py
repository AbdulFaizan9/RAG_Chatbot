"""Tiny deterministic game environments used by the RL projects."""
from __future__ import annotations

import numpy as np


class Catch:
    """Catch the falling ball. Actions: 0=left, 1=stay, 2=right. Reward +1 catch / -1 miss."""

    n_actions = 3

    def __init__(self, width: int = 5, height: int = 6, seed: int | None = None):
        if width < 2 or height < width:
            raise ValueError("need width >= 2 and height >= width so every ball is catchable")
        self.width, self.height = width, height
        self.rng = np.random.default_rng(seed)
        self.obs_dim = 2 * width + 1
        self.reset()

    def reset(self) -> np.ndarray:
        self.ball_col = int(self.rng.integers(self.width))
        self.ball_row = 0
        self.paddle = int(self.rng.integers(self.width))
        return self._obs()

    def _obs(self) -> np.ndarray:
        o = np.zeros(self.obs_dim)
        o[self.ball_col] = 1.0
        o[self.width + self.paddle] = 1.0
        o[-1] = self.ball_row / (self.height - 1)
        return o

    def step(self, action: int):
        if action not in (0, 1, 2):
            raise ValueError(f"invalid action {action}")
        self.paddle = int(np.clip(self.paddle + action - 1, 0, self.width - 1))
        self.ball_row += 1
        done = self.ball_row >= self.height - 1
        reward = 0.0
        if done:
            reward = 1.0 if self.paddle == self.ball_col else -1.0
        return self._obs(), reward, done


class GridWorld:
    """5x5 grid: start top-left, goal bottom-right, pits and a wall. Actions: 0 up 1 down 2 left 3 right."""

    n_actions = 4
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, size: int = 5, max_steps: int = 40):
        self.size, self.max_steps = size, max_steps
        self.obs_dim = size * size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.pits = {(1, 3), (3, 1)}
        self.walls = {(2, 2)}
        self.reset()

    def reset(self) -> np.ndarray:
        self.pos = self.start
        self.t = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        o = np.zeros(self.obs_dim)
        o[self.pos[0] * self.size + self.pos[1]] = 1.0
        return o

    def step(self, action: int):
        if action not in range(4):
            raise ValueError(f"invalid action {action}")
        dr, dc = self.MOVES[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.walls:
            self.pos = (nr, nc)
        self.t += 1
        if self.pos == self.goal:
            return self._obs(), 1.0, True
        if self.pos in self.pits:
            return self._obs(), -1.0, True
        return self._obs(), -0.02, self.t >= self.max_steps
