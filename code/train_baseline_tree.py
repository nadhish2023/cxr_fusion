import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import config


BASELINE_DIR = config.CHECKPOINT_DIR / "baselines"
CHECKPOINT_NAMES = {
    "logistic_regression": "baseline_logistic_regression.joblib",
    "random_forest": "baseline_random_forest.joblib",
    "xgboost": "baseline_xgboost.joblib",
}


def load_training_data():
    rows = pd.read_csv(config.SPLIT_DIR / "train.csv")
    features = rows[config.TABULAR_FEATURE_COLUMNS].to_numpy(dtype="float32")
    labels = rows[config.LABEL_COLUMNS].to_numpy(dtype="int32")
    return features, labels


def train_random_forest(features, labels):
    model = OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=config.SEED,
            n_jobs=-1,
        ),
        n_jobs=-1,
    )
    model.fit(features, labels)
    return model


def train_logistic_regression(features, labels):
    model = OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=config.SEED,
        ),
        n_jobs=-1,
    )
    model.fit(features, labels)
    return model


def train_xgboost(features, labels):
    from xgboost import XGBClassifier

    models = []
    for label_index in range(labels.shape[1]):
        positives = labels[:, label_index].sum()
        negatives = len(labels) - positives
        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=float(negatives / max(positives, 1)),
            tree_method="hist",
            random_state=config.SEED,
            n_jobs=-1,
        )
        model.fit(features, labels[:, label_index])
        models.append(model)
    return models


def train_model(model_name, features, labels):
    if model_name == "logistic_regression":
        model = train_logistic_regression(features, labels)
    elif model_name == "random_forest":
        model = train_random_forest(features, labels)
    elif model_name == "xgboost":
        model = train_xgboost(features, labels)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint_path = BASELINE_DIR / CHECKPOINT_NAMES[model_name]
    joblib.dump(
        {
            "model": model,
            "feature_columns": config.TABULAR_FEATURE_COLUMNS,
            "label_columns": config.LABEL_COLUMNS,
            "model_name": model_name,
        },
        checkpoint_path,
    )
    print(f"Saved {model_name} baseline to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train non-neural clinical baselines.")
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest", "xgboost", "all"],
        default="all",
        help="Non-neural clinical baseline to train.",
    )
    args = parser.parse_args()

    np.random.seed(config.SEED)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    features, labels = load_training_data()
    models = list(CHECKPOINT_NAMES) if args.model == "all" else [args.model]

    for model_name in models:
        print(f"Training {model_name} clinical baseline...")
        train_model(model_name, features, labels)


if __name__ == "__main__":
    main()