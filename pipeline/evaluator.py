"""Évaluation d'un modèle entraîné sur des données de test."""
from utils import metrics


class Evaluator:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def evaluate(self, X, y):
        proba = self.model.predict(X)
        y_pred = (proba >= self.threshold).astype(int)
        return {
            "loss": metrics.binary_cross_entropy(y, proba),
            "accuracy": metrics.accuracy(y, y_pred),
            "precision": metrics.precision(y, y_pred),
            "recall": metrics.recall(y, y_pred),
            "f1": metrics.f1_score(y, y_pred),
            "confusion": metrics.confusion_matrix(y, y_pred),
        }

    @staticmethod
    def format_report(result):
        cm = result["confusion"]
        return (
            f"  Perte (test) : {result['loss']:.4f}\n"
            f"  Exactitude   : {result['accuracy']:.1%}\n"
            f"  Précision    : {result['precision']:.1%}\n"
            f"  Rappel       : {result['recall']:.1%}\n"
            f"  F1-score     : {result['f1']:.3f}\n"
            f"  Matrice      : VP={cm['tp']} VN={cm['tn']} FP={cm['fp']} FN={cm['fn']}"
        )
