import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from main_model.code import config


CHECKPOINT_PATH = config.BASELINE_CHECKPOINT_DIR / "baseline_xgboost.joblib"


def main():
    train_rows = pd.read_csv(config.SPLIT_DIR / "train.csv")
    features = train_rows[config.TABULAR_FEATURE_COLUMNS].to_numpy(dtype="float32")
    labels = train_rows[config.LABEL_COLUMNS].to_numpy(dtype="float32")
    models = []

    for index, label in enumerate(config.LABEL_COLUMNS):
        valid = np.isfinite(labels[:, index])
        target = labels[valid, index].astype("int32")
        positives = int(target.sum())
        negatives = int(len(target) - positives)
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
        model.fit(features[valid], target)
        models.append(model)
        print(f"Trained XGBoost for {label}: {int(valid.sum()):,} valid rows")

    config.BASELINE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "feature_columns": config.TABULAR_FEATURE_COLUMNS,
            "label_columns": config.LABEL_COLUMNS,
            "model_name": "xgboost",
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved XGBoost baseline to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
