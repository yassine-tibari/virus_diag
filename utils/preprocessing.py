"""Prétraitement des données patients."""
import numpy as np


class StandardScaler:
    """Standardise les colonnes : (x - moyenne) / écart-type."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler doit être ajusté (fit) avant transform.")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def fill_missing(X):
    """Remplace les valeurs manquantes (NaN) par la moyenne de leur colonne."""
    X = np.array(X, dtype=float)
    col_means = np.nanmean(X, axis=0)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = col_means[cols]
    return X


def train_test_split(X, y, test_size=0.2, seed=42):
    """Mélange puis sépare les données en ensembles d'entraînement et de test."""
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
