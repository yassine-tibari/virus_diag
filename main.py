"""Point d'entrée du projet virus_diag.

Usage (depuis le dossier virus_diag) :
    python main.py                      # ouvre l'interface graphique (entraîne si besoin)
    python main.py train                # entraîne, évalue et sauvegarde le modèle
    python main.py train --model neural # idem avec le réseau de neurones
    python main.py cli                  # interface clinique en ligne de commande
    python main.py serve --port 8000    # API HTTP
"""
import argparse
import os

from app import interface_clinique
from app.api import DiagnosisAPI
from app.clinical_predictor import ClinicalPredictor
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.optimizer import SGD, Adam
from pipeline.evaluator import Evaluator
from pipeline.trainer import Trainer
from utils.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "patient_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "clinical_predictor.pkl")


def build_model_and_optimizer(name, n_features):
    if name == "neural":
        return NeuralNetwork(n_features, hidden_units=8), Adam(lr=0.02)
    return LogisticRegression(n_features), SGD(lr=0.5)


DEFAULT_EPOCHS = {"logistic": 500, "neural": 150}  # le réseau sur-apprend vite


def train(model_name="logistic", epochs=None, save_path=MODEL_PATH):
    """Pipeline complet : données -> prétraitement -> entraînement -> évaluation -> sauvegarde."""
    epochs = epochs or DEFAULT_EPOCHS[model_name]
    dataset = Dataset.from_csv(DATA_PATH)
    print(f"Données : {len(dataset)} patients, variables : {', '.join(dataset.feature_names)}")

    X_train, X_test, y_train, y_test = dataset.split(test_size=0.2, seed=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # ajusté sur l'entraînement uniquement
    X_test = scaler.transform(X_test)

    model, optimizer = build_model_and_optimizer(model_name, len(dataset.feature_names))
    print(f"\nEntraînement ({type(model).__name__}, {type(optimizer).__name__}) :")
    Trainer(model, optimizer, epochs=epochs).train(X_train, y_train)

    print("\nÉvaluation sur le jeu de test :")
    print(Evaluator.format_report(Evaluator(model).evaluate(X_test, y_test)))

    predictor = ClinicalPredictor(model, scaler, dataset.feature_names)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    predictor.save(save_path)
    print(f"\nPrédicteur sauvegardé : {save_path}")
    return predictor


def load_or_train(model_name="logistic", epochs=None):
    if os.path.exists(MODEL_PATH):
        return ClinicalPredictor.load(MODEL_PATH)
    print("Aucun modèle sauvegardé : entraînement en cours...\n")
    return train(model_name, epochs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="virus_diag - diagnostic assisté par IA")
    parser.add_argument("command", nargs="?", default="gui",
                        choices=["gui", "train", "cli", "serve"])
    parser.add_argument("--model", choices=["logistic", "neural"], default="logistic")
    parser.add_argument("--epochs", type=int, default=None,
                        help="500 (logistic) ou 150 (neural) par défaut")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.command == "train":
        train(args.model, args.epochs)
    elif args.command == "gui":
        predictor = load_or_train(args.model, args.epochs)
        try:
            interface_clinique.run_gui(predictor)
        except RuntimeError as exc:
            print(f"\n{exc}\nBascule sur l'interface en ligne de commande.\n")
            interface_clinique.run(predictor)
    elif args.command == "cli":
        interface_clinique.run(load_or_train(args.model, args.epochs))
    else:
        DiagnosisAPI(load_or_train(args.model, args.epochs)).serve(args.host, args.port)


if __name__ == "__main__":
    main()
