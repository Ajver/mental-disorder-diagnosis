import pickle

import numpy as np
import pandas as pd
from fastmlapi import MLController, preprocessing, postprocessing

DATA_COLUMNS = pd.read_csv("../dataset/dataset.csv").columns.drop(["Patient Number", "Expert Diagnose"]).values.tolist()
print("data columns:", DATA_COLUMNS, sep="\n")


class MentalDisordersClassifier(MLController):
    model_name = "mental-disorders-classifier"
    model_version = "1.0.0"

    def load_model(self):
        with open("../models/logistic_regression.pkl", "rb") as f:
            model = pickle.load(f)
            return model

    @preprocessing
    def preprocess(self, data: dict) -> pd.DataFrame:
        # Wrapping with DataFrame to filter only expected columns and ensure correct order
        df = pd.DataFrame(columns=DATA_COLUMNS, data=[data])
        return df

    @postprocessing
    def postprocess(self, prediction: np.ndarray) -> dict:
        response = {
            "diagnose": prediction[0]
        }
        return response


if __name__ == "__main__":
    MentalDisordersClassifier().run()
