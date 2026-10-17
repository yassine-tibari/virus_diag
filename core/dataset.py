"""Chargement et découpage du jeu de données patients."""
import csv

import numpy as np

from utils.preprocessing import fill_missing, train_test_split


class Dataset:
    """Contient les caractéristiques (X), les étiquettes (y) et les noms de colonnes."""

    def __init__(self, X, y, feature_names):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.feature_names = list(feature_names)

    @classmethod
    def from_csv(cls, path, target="infected"):
        """Lit un CSV ; les cases vides sont remplacées par la moyenne de la colonne."""
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            feature_names = [c for c in reader.fieldnames if c != target]
            rows = list(reader)
        X = [
            [float(r[c]) if r[c].strip() != "" else np.nan for c in feature_names]
            for r in rows
        ]
        y = [float(r[target]) for r in rows]
        return cls(fill_missing(X), y, feature_names)

    def __len__(self):
        return len(self.y)

    def split(self, test_size=0.2, seed=42):
        """Retourne X_train, X_test, y_train, y_test."""
        return train_test_split(self.X, self.y, test_size=test_size, seed=seed)
