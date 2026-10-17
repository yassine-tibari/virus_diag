"""Régression logistique implémentée avec numpy."""
import numpy as np

from core.model import Model, sigmoid
from utils.metrics import binary_cross_entropy


class LogisticRegression(Model):
    def __init__(self, n_features, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.params = {"W": rng.normal(0, 0.01, n_features), "b": np.zeros(1)}

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return sigmoid(X @ self.params["W"] + self.params["b"])

    def loss_and_grads(self, X, y):
        p = self.predict(X)
        err = p - y
        n = len(y)
        grads = {"W": X.T @ err / n, "b": np.array([err.mean()])}
        return binary_cross_entropy(y, p), grads
