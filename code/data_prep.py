import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def validate_columns(df):
    required = {"subject_id", "image_path", "gender", *config.LABEL_COLUMNS, *config.TABULAR_NUMERIC_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        fail(f"CSV is missing required columns: {', '.join(missing)}")


def validate_images(df):
    missing = []
    root = Path(config.IMAGE_ROOT).expanduser().resolve()
    for image_path in df["image_path"].fillna("").astype(str):
        candidate = (root / image_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            missing.append(f"{image_path} (outside IMAGE_ROOT)")
            continue
        if not candidate.is_file():
            missing.append(image_path)
            if len(missing) >= 20:
                break
    if missing:
        suffix = "" if len(missing) < 20 else " (showing first 20)"
        fail(f"Missing image files under {root}: {', '.join(missing)}{suffix}")


def prepare_labels(df):
    for label in config.LABEL_COLUMNS:
        values = pd.to_numeric(df[label], errors="coerce").fillna(0)
        values = values.where(values != -1, 1)
        if label == "Pneumonia":
            values = pd.to_numeric(df[label], errors="coerce").fillna(0).where(df[label] != -1, 0)
        df[label] = values.clip(0, 1).astype(np.float32)


def build_features(df, stats=None):
    result = df.copy()
    for column, (lower, upper) in config.CLIP_RANGES.items():
        result[column] = pd.to_numeric(result[column], errors="coerce").clip(lower, upper)
        result[f"{column}_missing"] = result[column].isna().astype(np.float32)
    result["age_at_imaging"] = pd.to_numeric(result["age_at_imaging"], errors="coerce")

    if stats is None:
        stats = {
            "means": {column: float(result[column].mean()) for column in config.TABULAR_NUMERIC_COLUMNS},
            "stds": {column: float(result[column].std()) if result[column].std() > 0 else 1.0 for column in config.TABULAR_NUMERIC_COLUMNS},
        }
    for column in config.TABULAR_NUMERIC_COLUMNS:
        result[column] = result[column].fillna(stats["means"][column])
        result[column] = (result[column] - stats["means"][column]) / stats["stds"][column]

    gender = result["gender"].fillna("unknown").astype(str).str.upper()
    result["gender_M"] = (gender == "M").astype(np.float32)
    result["gender_F"] = (gender == "F").astype(np.float32)
    result["gender_unknown"] = (~gender.isin(["M", "F"])).astype(np.float32)
    return result, stats


def main():
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    csv_path = Path(config.CSV_PATH).expanduser().resolve()
    if not csv_path.is_file():
        fail(f"CSV file not found: {csv_path}. Update CSV_PATH in config.py.")
    image_root = Path(config.IMAGE_ROOT).expanduser().resolve()
    if not image_root.is_dir():
        fail(f"Image root not found: {image_root}. Update IMAGE_ROOT in config.py.")

    df = pd.read_csv(csv_path)
    validate_columns(df)
    if df["subject_id"].isna().any():
        fail("CSV contains rows with missing subject_id.")
    validate_images(df)
    prepare_labels(df)

    subjects = df["subject_id"].drop_duplicates().to_numpy()
    train_subjects, heldout_subjects = train_test_split(subjects, test_size=0.30, random_state=config.SEED)
    val_subjects, test_subjects = train_test_split(heldout_subjects, test_size=0.50, random_state=config.SEED)
    subject_sets = {"train": set(train_subjects), "val": set(val_subjects), "test": set(test_subjects)}
    if set(train_subjects) & set(val_subjects) or set(train_subjects) & set(test_subjects) or set(val_subjects) & set(test_subjects):
        fail("Subject leakage detected while creating splits.")

    train_raw = df[df["subject_id"].isin(subject_sets["train"])]
    _, stats = build_features(train_raw)
    processed, _ = build_features(df, stats)
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for name, subject_ids in subject_sets.items():
        split = processed[processed["subject_id"].isin(subject_ids)]
        split.to_csv(config.SPLIT_DIR / f"{name}.csv", index=False)
    with open(config.PREPROCESSING_PATH, "w", encoding="utf-8") as handle:
        json.dump({"seed": config.SEED, "image_root": str(image_root), **stats}, handle, indent=2)
    print(f"Prepared {len(processed):,} rows and {len(subjects):,} subjects.")
    print(f"Splits: train={len(train_raw):,}, val={len(processed[processed.subject_id.isin(subject_sets['val'])]):,}, test={len(processed[processed.subject_id.isin(subject_sets['test'])]):,}")
    print(f"Saved split files to {config.SPLIT_DIR}")


if __name__ == "__main__":
    main()
