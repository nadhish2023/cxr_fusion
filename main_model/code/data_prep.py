import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    from . import config
except ImportError:
    import config


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def validate_columns(df):
    required = {"subject_id", "image_path", "gender", *config.LABEL_COLUMNS, *config.TABULAR_NUMERIC_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        fail(f"CSV is missing required columns: {', '.join(missing)}")


def clean_images(df):
    invalid_rows = []
    root = Path(config.IMAGE_ROOT).expanduser().resolve()
    for index, image_path in df["image_path"].fillna("").astype(str).items():
        relative_path = image_path.removeprefix("files/")
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
            with Image.open(candidate) as image:
                image.load()
        except (OSError, ValueError) as error:
            invalid_rows.append({
                "row_index": int(index),
                "image_path": image_path,
                "reason": str(error),
            })

    if invalid_rows:
        report_path = config.METRICS_DIR / "invalid_images.csv"
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(invalid_rows).to_csv(report_path, index=False)
        print(f"Removed {len(invalid_rows):,} rows with missing or unreadable images.")
        print(f"Invalid-image report: {report_path}")
        df = df.drop(index=[item["row_index"] for item in invalid_rows])

    if df.empty:
        fail("No usable rows remain after image validation.")
    return df.reset_index(drop=True)


def prepare_labels(df):
    label_audit = {}
    for label in config.LABEL_COLUMNS:
        values = pd.to_numeric(df[label], errors="coerce")
        definite = values.isin([0, 1])
        label_audit[label] = {
            "definite_negative": int((values == 0).sum()),
            "definite_positive": int((values == 1).sum()),
            "uncertain": int((values == -1).sum()),
            "missing_or_invalid": int((~values.notna() | ~values.isin([-1, 0, 1])).sum()),
        }
        df[label] = values.where(definite).astype(np.float32)
    label_audit["rows_removed_by_policy"] = 0
    label_audit["masked_label_values"] = int(sum(
        item["uncertain"] + item["missing_or_invalid"]
        for key, item in label_audit.items()
        if key in config.LABEL_COLUMNS
    ))
    return df, label_audit


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
    df = clean_images(df)
    df, label_audit = prepare_labels(df)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_DIR / "label_audit.json", "w", encoding="utf-8") as handle:
        json.dump({"policy": config.LABEL_POLICY, "labels": label_audit}, handle, indent=2)

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
