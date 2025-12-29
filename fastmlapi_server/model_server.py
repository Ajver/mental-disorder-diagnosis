import numpy as np
from fastmlapi import MLController, preprocessing, postprocessing


class MyClassifier(MLController):
    model_name = "my-classifier"
    model_version = "1.0.0"

    def load_model(self):
        # TODO
        return None

    @preprocessing
    def preprocess(self, data: dict) -> np.ndarray:
        return np.array(data["features"]).reshape(1, -1)

    @postprocessing
    def postprocess(self, prediction: np.ndarray) -> dict:
        return {"class": int(prediction[0])}


if __name__ == "__main__":
    MyClassifier().run()
