"""Classe de base commune à tous les modèles (interface polymorphe)."""
import pickle
from abc import ABC, abstractmethod

import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class Model(ABC):
    """Tout modèle expose predict() et loss_and_grads().

    - predict(X) renvoie des probabilités d'infection entre 0 et 1.
    - loss_and_grads(X, y) renvoie (perte, gradients) pour l'entraînement.
    - params est un dictionnaire de tableaux numpy mis à jour par l'optimiseur.
    """

    def __init__(self):
        self.params = {}

    @abstractmethod
    def predict(self, X):
        """Retourne un tableau 1D de probabilités."""

    @abstractmethod
    def loss_and_grads(self, X, y):
        """Retourne (perte, dict de gradients de même clés que params)."""

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
