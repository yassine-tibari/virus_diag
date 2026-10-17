"""ClinicalPredictor : réponse à l'exercice (question 1).

Fonctionne avec N'IMPORTE QUEL modèle possédant une méthode predict()
(polymorphisme / duck typing) : il ne connaît rien à l'entraînement.
"""
import pickle

import numpy as np


class ClinicalPredictor:
    LABEL_INFECTED = "Infecté"
    LABEL_HEALTHY = "Sain"

    def __init__(self, model, preprocessor=None, feature_names=None, threshold=0.5):
        if not hasattr(model, "predict"):
            raise TypeError("Le modèle doit posséder une méthode predict().")
        self.model = model                  # modèle déjà entraîné
        self.preprocessor = preprocessor    # ex. StandardScaler déjà ajusté
        self.feature_names = list(feature_names) if feature_names else None
        self.threshold = threshold

    def _to_array(self, patient_data):
        """Accepte un dict {nom: valeur} ou une liste/tableau de valeurs."""
        if isinstance(patient_data, dict):
            if not self.feature_names:
                raise ValueError("feature_names requis pour utiliser un dictionnaire.")
            missing = [n for n in self.feature_names if n not in patient_data]
            if missing:
                raise ValueError(f"Champs manquants : {', '.join(missing)}")
            values = [float(patient_data[n]) for n in self.feature_names]
        else:
            values = [float(v) for v in patient_data]
        x = np.asarray(values, dtype=float).reshape(1, -1)
        if self.preprocessor is not None:
            x = self.preprocessor.transform(x)
        return x

    def predict_proba(self, patient_data):
        """Probabilité d'infection renvoyée par le modèle (float)."""
        prediction = self.model.predict(self._to_array(patient_data))
        return float(np.ravel(prediction)[0])

    def label(self, probability):
        return self.LABEL_INFECTED if probability >= self.threshold else self.LABEL_HEALTHY

    def diagnose(self, patient_data):
        """'Infecté' si prédiction >= 0.5, 'Sain' sinon."""
        return self.label(self.predict_proba(patient_data))

    # --- persistance (modèle + prétraitement dans un seul fichier) ---
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
