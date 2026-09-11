import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


MIN_GROUP_SIZE = 30
PREVALENCE_IMBALANCE_LIMIT = 0.10
MISSINGNESS_GAP_LIMIT = 0.10
SPLIT_DRIFT_LIMIT = 0.05


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def safe_float(value):
    return None if pd.isna(value) else float(value)


def label_values(series, label):
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if label == "Pneumonia":
        return values.where(series != -1, 0).clip(0, 1)
    return values.where(values != -1, 1).clip(0, 1)


def group_rates(df, group_column, value_column):
    result = {}
    for group, rows in df.groupby(group_column, dropna=False):
        key = "unknown" if pd.isna(group) else str(group)
        result[key] = {
            "n": int(len(rows)),
            "rate": safe_float(rows[value_column].mean()),
        }
    return result


def describe_clinical(df):
    summary = {
        "rows": int(len(df)),
        "unique_subjects": int(df["subject_id"].nunique()),
        "images_per_subject": {
            "mean": safe_float(df.groupby("subject_id").size().mean()),
            "median": safe_float(df.groupby("subject_id").size().median()),
            "max": int(df.groupby("subject_id").size().max()),
        },
        "gender_counts": {str(key): int(value) for key, value in df["gender"].fillna("unknown").astype(str).str.upper().value_counts().items()},
        "numeric": {},
        "missingness": {},
        "label_prevalence": {},
    }
    for column in config.TABULAR_NUMERIC_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        summary["numeric"][column] = {
            "count": int(values.notna().sum()),
            "missing": int(values.isna().sum()),
            "missing_rate": safe_float(values.isna().mean()),
            "mean": safe_float(values.mean()),
            "median": safe_float(values.median()),
            "std": safe_float(values.std()),
            "min": safe_float(values.min()),
            "max": safe_float(values.max()),
        }
    for column in config.VITAL_COLUMNS:
        summary["missingness"][column] = {
            "overall_rate": safe_float(pd.to_numeric(df[column], errors="coerce").isna().mean()),
            "by_gender": group_rates(df.assign(_missing=pd.to_numeric(df[column], errors="coerce").isna()), "gender", "_missing"),
        }
    for label in config.LABEL_COLUMNS:
        df = df.copy()
        df["_label"] = label_values(df[label], label)
        summary["label_prevalence"][label] = {
            "positive": int(df["_label"].sum()),
            "negative": int((1 - df["_label"]).sum()),
            "rate": safe_float(df["_label"].mean()),
            "by_gender": group_rates(df.assign(_label=df["_label"]), "gender", "_label"),
        }
    return summary


def detect_signals(df, split_dir):
    signals = []
    checks = {}
    gender = df["gender"].fillna("unknown").astype(str).str.upper()
    gender_share = gender.value_counts(normalize=True)
    checks["gender_representation"] = {str(key): float(value) for key, value in gender_share.items()}
    if (gender_share < 0.10).any():
        signals.append("A gender category represents less than 10% of rows; group-specific performance may be unstable.")
    if (gender == "UNKNOWN").mean() > 0.05:
        signals.append("More than 5% of gender values are unknown or outside M/F.")

    for label in config.LABEL_COLUMNS:
        df = df.copy()
        df["_label"] = label_values(df[label], label)
        overall = float(df["_label"].mean())
        checks.setdefault("gender_label_gap", {})[label] = {}
        rates = df.assign(_gender=gender).groupby("_gender")["_label"].agg(["mean", "count"])
        eligible = rates[rates["count"] >= MIN_GROUP_SIZE]
        if len(eligible) >= 2:
            gap = float(eligible["mean"].max() - eligible["mean"].min())
            checks["gender_label_gap"][label] = {"overall": overall, "gap": gap, "groups": eligible["mean"].to_dict()}
            if gap >= PREVALENCE_IMBALANCE_LIMIT:
                signals.append(f"{label} prevalence differs by at least {PREVALENCE_IMBALANCE_LIMIT:.0%} between eligible gender groups.")

    for column in config.VITAL_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        missing_by_gender = df.assign(_missing=values.isna(), _gender=gender).groupby("_gender")["_missing"].agg(["mean", "count"])
        eligible = missing_by_gender[missing_by_gender["count"] >= MIN_GROUP_SIZE]
        gap = float(eligible["mean"].max() - eligible["mean"].min()) if len(eligible) >= 2 else 0.0
        checks.setdefault("missingness_gender_gap", {})[column] = {"max_gap": gap, "groups": eligible["mean"].to_dict()}
        if gap >= MISSINGNESS_GAP_LIMIT:
            signals.append(f"{column} missingness differs by at least {MISSINGNESS_GAP_LIMIT:.0%} between eligible gender groups.")

    for column, (lower, upper) in config.CLIP_RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce")
        out_of_range = ((values < lower) | (values > upper)).mean()
        checks.setdefault("out_of_range_rates", {})[column] = float(out_of_range)
        if out_of_range > 0.01:
            signals.append(f"More than 1% of {column} values are outside the configured clinical range.")

    split_prevalence = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.csv"
        if not path.is_file():
            continue
        split = pd.read_csv(path)
        split_prevalence[name] = {label: float(pd.to_numeric(split[label], errors="coerce").mean()) for label in config.LABEL_COLUMNS}
    checks["split_label_prevalence"] = split_prevalence
    if len(split_prevalence) == 3:
        for label in config.LABEL_COLUMNS:
            rates = [values[label] for values in split_prevalence.values()]
            if max(rates) - min(rates) >= SPLIT_DRIFT_LIMIT:
                signals.append(f"{label} prevalence differs by at least {SPLIT_DRIFT_LIMIT:.0%} across train/validation/test.")

    subject_counts = df.groupby("subject_id").size()
    checks["subject_concentration"] = {
        "max_images_per_subject": int(subject_counts.max()),
        "subjects_with_more_than_10_images": int((subject_counts > 10).sum()),
        "top_subject_row_share": float(subject_counts.max() / len(df)),
    }
    if subject_counts.max() / len(df) > 0.01:
        signals.append("One subject contributes more than 1% of all rows; subject-level splitting is especially important.")
    return signals, checks


def print_report(report):
    summary = report["clinical_summary"]
    print(f"Rows: {summary['rows']:,}")
    print(f"Unique subjects: {summary['unique_subjects']:,}")
    print("\nLabel prevalence:")
    for label, values in summary["label_prevalence"].items():
        print(f"  {label}: {values['rate']:.3%} positive ({values['positive']:,}/{summary['rows']:,})")
    print("\nClinical numeric summary:")
    for column, values in summary["numeric"].items():
        print(f"  {column}: mean={values['mean']:.3f}, median={values['median']:.3f}, missing={values['missing_rate']:.3%}")
    print("\nGender counts:")
    print(f"  {summary['gender_counts']}")
    print("\nBias and dataset-quality screening signals:")
    if report["signals"]:
        for signal in report["signals"]:
            print(f"  WARNING: {signal}")
    else:
        print("  No configured heuristic signal exceeded its warning threshold.")
    print("  These are screening signals, not proof of unfairness or causality.")


def main():
    parser = argparse.ArgumentParser(description="Describe clinical data and screen for dataset bias signals.")
    parser.add_argument("--csv", default=str(config.CSV_PATH))
    parser.add_argument("--split-dir", default=str(config.SPLIT_DIR))
    parser.add_argument("--output", default=str(config.METRICS_DIR / "clinical_data_audit.json"))
    args = parser.parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    split_dir = Path(args.split_dir).expanduser().resolve()
    if not csv_path.is_file():
        fail(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"subject_id", "gender", *config.LABEL_COLUMNS, *config.TABULAR_NUMERIC_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        fail(f"CSV is missing columns: {', '.join(missing)}")
    report = {
        "csv": str(csv_path),
        "clinical_summary": describe_clinical(df),
    }
    report["signals"], report["checks"] = detect_signals(df, split_dir)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=lambda value: value.item() if hasattr(value, "item") else value)
    print_report(report)
    print(f"\nSaved audit report to {output_path}")


if __name__ == "__main__":
    main()
