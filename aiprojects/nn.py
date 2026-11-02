"""A small fully-connected neural network with manual back-prop and Adam (NumPy only)."""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


class MLP:
    """ReLU MLP with a linear output layer.

    ``forward`` returns the output and a cache; ``backward`` turns dLoss/dOutput into
    parameter gradients; ``step`` applies Adam.
    """

    def __init__(self, sizes: list[int], rng: np.random.Generator | None = None):
        if len(sizes) < 2:
            raise ValueError("need at least input and output sizes")
        self.rng = rng or np.random.default_rng(0)
        self.sizes = list(sizes)
        self.W = [self.rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0

    def forward(self, x: np.ndarray):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        acts, pre = [x], []
        last = len(self.W) - 1
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ w + b
            pre.append(z)
            acts.append(z if i == last else np.maximum(z, 0.0))
        return acts[-1], (acts, pre)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def backward(self, cache, grad_out: np.ndarray):
        acts, pre = cache
        delta = np.asarray(grad_out, dtype=float)
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            dW[i] = acts[i].T @ delta
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (pre[i - 1] > 0)
        return dW, db

    def step(self, grads, lr: float = 1e-3, clip: float | None = 5.0, b1=0.9, b2=0.999, eps=1e-8):
        dW, db = grads
        params = self.W + self.b
        gs = list(dW) + list(db)
        if clip is not None:
            total = np.sqrt(sum(float((g ** 2).sum()) for g in gs))
            if total > clip:
                gs = [g * (clip / total) for g in gs]
        self._t += 1
        for k, (p, g) in enumerate(zip(params, gs)):
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mh = self._m[k] / (1 - b1 ** self._t)
            vh = self._v[k] / (1 - b2 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + eps)  # in-place update of the parameter array

    def copy_from(self, other: "MLP") -> None:
        for i in range(len(self.W)):
            self.W[i][...] = other.W[i]
            self.b[i][...] = other.b[i]
