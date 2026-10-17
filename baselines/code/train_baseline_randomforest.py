import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from main_model.code import config


CHECKPOINT_PATH = config.BASELINE_CHECKPOINT_DIR / "baseline_randomforest.joblib"


def main():
    train_rows = pd.read_csv(config.SPLIT_DIR / "train.csv")
    features = train_rows[config.TABULAR_FEATURE_COLUMNS].to_numpy(dtype="float32")
    labels = train_rows[config.LABEL_COLUMNS].to_numpy(dtype="float32")
    models = []

    for index, label in enumerate(config.LABEL_COLUMNS):
        valid = np.isfinite(labels[:, index])
        model = RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=config.SEED,
            n_jobs=-1,
        )
        model.fit(features[valid], labels[valid, index].astype("int32"))
        models.append(model)
        print(f"Trained Random Forest for {label}: {int(valid.sum()):,} valid rows")

    config.BASELINE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "feature_columns": config.TABULAR_FEATURE_COLUMNS,
            "label_columns": config.LABEL_COLUMNS,
            "model_name": "randomforest",
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved Random Forest baseline to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
