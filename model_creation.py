import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_random_forest(params: dict, random_state: int = 42) -> Pipeline:
    n_features_to_select = params["n_features_to_select"]
    n_estimators = params["n_estimators"]
    min_samples_split = params["min_samples_split"]
    max_depth = params["max_depth"]

    scaler = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
    }[params["scaler_name"]]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        random_state=random_state,
        max_depth=max_depth
    )
    pipe = Pipeline([
        ("scaler", scaler),
        ("selector", RFE(model, n_features_to_select=n_features_to_select)),
        ("classifier", model),
    ])
    return pipe


def create_knn(params: dict) -> Pipeline:
    n_features_to_select = params["n_features_to_select"]
    n_neighbors = params["n_neighbors"]
    weights = params["weights"]
    metric = params["metric"]

    scaler = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
    }[params["scaler_name"]]

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
    )

    pipe = Pipeline([
        ("scaler", scaler),
        ("selector", SelectKBest(k=n_features_to_select)),
        ("classifier", model),
    ])
    return pipe


def create_logistic_regression(params: dict, random_state: int = 42) -> Pipeline:
    n_features_to_select = params["n_features_to_select"]
    C = params["C"]
    solver = params["solver"]
    max_iter = params["max_iter"]

    scaler = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
    }[params["scaler_name"]]

    ovr_model = OneVsRestClassifier(LogisticRegression(
        C=C, solver=solver,
        max_iter=max_iter,
        random_state=random_state)
    )

    pipe = Pipeline([
        ("scaler", scaler),
        ("selector", RFE(LogisticRegression(C=C, max_iter=max_iter, random_state=random_state), n_features_to_select=n_features_to_select)),
        ("classifier", ovr_model),
    ])
    return pipe


def save_model_params(model_name: str, params: dict):
    with open(f"model_params/{model_name}.json", "w") as f:
        params_str = json.dumps(params, indent=4)
        f.write(params_str)


def load_model_params(model_name: str) -> dict:
    with open(f"model_params/{model_name}.json", "r") as f:
        params_str = f.read()
        params = json.loads(params_str)
        return params
