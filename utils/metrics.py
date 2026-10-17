"""Métriques d'évaluation pour la classification binaire."""
import numpy as np

_EPS = 1e-12


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def binary_cross_entropy(y_true, y_prob):
    """Perte d'entropie croisée binaire (utilisée par les modèles et l'évaluateur)."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), _EPS, 1 - _EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def confusion_matrix(y_true, y_pred):
    """Retourne les comptes vrais/faux positifs et négatifs."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return _safe_div(cm["tp"] + cm["tn"], sum(cm.values()))


def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return _safe_div(cm["tp"], cm["tp"] + cm["fp"])


def recall(y_true, y_pred):
    """Sensibilité : part des patients infectés correctement détectés."""
    cm = confusion_matrix(y_true, y_pred)
    return _safe_div(cm["tp"], cm["tp"] + cm["fn"])


def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return _safe_div(2 * p * r, p + r)
