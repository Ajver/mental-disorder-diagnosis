"""
Finally, the LogisticRegression model was selected,
because all models seemed to have similar accuracy, and the LogisticRegression
is the simples and the most interpretable among others.
"""

import os
import pickle

import pandas as pd

from src.model_creation import load_model_params, create_logistic_regression

best_params = load_model_params("logistic_regression")
model = create_logistic_regression(best_params)

TARGET_COL_NAME = "Expert Diagnose"
dataset = pd.read_csv("../dataset/dataset.csv")
X, y = dataset.drop(columns=[TARGET_COL_NAME, "Patient Number"], axis=1), dataset[TARGET_COL_NAME]
model.fit(X, y)

MODELS_DIR = "../models"
os.makedirs(MODELS_DIR, exist_ok=True)
with open(f"{MODELS_DIR}/logistic_regression.pkl", "wb") as f:
    pickle.dump(model, f)
