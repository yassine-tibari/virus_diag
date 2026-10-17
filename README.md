# virus_diag — Diagnostic assisté par IA (Chap 5 : POO)

Projet pédagogique illustrant la programmation orientée objet dans une architecture IA.
Seule dépendance : `numpy`.

## Arborescence

```
virus_diag/
├── data/
│   └── patient_data.csv          # jeu de données (400 patients, synthétique)
├── core/
│   ├── dataset.py                # Dataset : lecture CSV, découpage train/test
│   ├── model.py                  # Model : classe abstraite (predict, loss_and_grads)
│   ├── logistic_regression.py    # LogisticRegression(Model)
│   ├── neural_network.py         # NeuralNetwork(Model)
│   └── optimizer.py              # SGD, Adam
├── pipeline/
│   ├── trainer.py                # Trainer : entraîne un modèle
│   └── evaluator.py              # Evaluator : calcule les métriques
├── utils/
│   ├── preprocessing.py          # StandardScaler, valeurs manquantes, split
│   └── metrics.py                # accuracy, précision, rappel, F1, entropie croisée
├── app/
│   ├── clinical_predictor.py     # ClinicalPredictor (exercice, question 1)
│   ├── api.py                    # API HTTP /diagnose
│   └── interface_clinique.py     # Interface graphique (Tkinter) + version texte
├── main.py                       # Orchestre tout le pipeline
└── requirements.txt
```

## Comment les fichiers interagissent

```
patient_data.csv ─► Dataset ─► StandardScaler ─► Trainer(Model, Optimizer) ─► Evaluator
                                    │                     │
                                    └──────────┬──────────┘
                                               ▼
                                      ClinicalPredictor  (sauvegardé dans models/)
                                         ▲            ▲
                                  DiagnosisAPI   interface_clinique
```

- `Trainer` et `ClinicalPredictor` sont **séparés** : l'application ne fait que prédire.
- `ClinicalPredictor` accepte **n'importe quel modèle** ayant une méthode `predict()`
  (polymorphisme / duck typing) : `LogisticRegression` ou `NeuralNetwork`.

## Utilisation

```bash
pip install -r requirements.txt

python main.py                      # ouvre l'interface graphique (entraîne au 1er lancement)
python main.py train                # entraîne (régression logistique), évalue, sauvegarde
python main.py train --model neural # idem avec le réseau de neurones
python main.py cli                  # interface clinique en ligne de commande
python main.py serve --port 8000    # API HTTP
```

L'interface graphique utilise Tkinter (inclus avec Python ; sous Linux : `sudo apt install python3-tk`).
Elle propose un formulaire patient avec validation, le diagnostic coloré (Infecté / Sain),
une jauge de probabilité, des exemples prêts à l'emploi et l'historique des consultations.

Exemple d'appel de l'API :

```bash
curl -X POST http://127.0.0.1:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"age": 62, "temperature": 39.1, "heart_rate": 104, "oxygen_saturation": 92,
       "cough": 1, "fatigue": 1, "contact_with_case": 1}'
```

## Utiliser ClinicalPredictor directement

```python
from app.clinical_predictor import ClinicalPredictor

predictor = ClinicalPredictor.load("models/clinical_predictor.pkl")
print(predictor.diagnose([62, 39.1, 104, 92, 1, 1, 1]))   # "Infecté" ou "Sain"
```
