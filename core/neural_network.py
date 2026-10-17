"""Réseau de neurones à une couche cachée (ReLU) et sortie sigmoïde."""
import numpy as np

from core.model import Model, sigmoid
from utils.metrics import binary_cross_entropy


class NeuralNetwork(Model):
    def __init__(self, n_features, hidden_units=8, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / n_features), (n_features, hidden_units)),
            "b1": np.zeros(hidden_units),
            "W2": rng.normal(0, np.sqrt(2 / hidden_units), hidden_units),
            "b2": np.zeros(1),
        }

    def _forward(self, X):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.maximum(0, z1)
        out = sigmoid(a1 @ p["W2"] + p["b2"])
        return z1, a1, out

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return self._forward(X)[2]

    def loss_and_grads(self, X, y):
        z1, a1, out = self._forward(X)
        n = len(y)
        dz2 = (out - y) / n
        dz1 = np.outer(dz2, self.params["W2"]) * (z1 > 0)
        grads = {
            "W1": X.T @ dz1,
            "b1": dz1.sum(axis=0),
            "W2": a1.T @ dz2,
            "b2": np.array([dz2.sum()]),
        }
        return binary_cross_entropy(y, out), grads
