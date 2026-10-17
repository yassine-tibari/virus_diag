"""Optimiseurs : mettent à jour les paramètres d'un modèle à partir des gradients."""
from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def step(self, params, grads):
        """Met à jour params (en place) avec grads."""


class SGD(Optimizer):
    """Descente de gradient avec momentum optionnel."""

    def __init__(self, lr=0.1, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, params, grads):
        for name, g in grads.items():
            v = self.velocity.get(name, np.zeros_like(params[name]))
            v = self.momentum * v - self.lr * g
            self.velocity[name] = v
            params[name] += v


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for name, g in grads.items():
            m = self.m.get(name, np.zeros_like(params[name]))
            v = self.v.get(name, np.zeros_like(params[name]))
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g ** 2
            self.m[name], self.v[name] = m, v
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
