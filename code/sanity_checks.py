import argparse
from pathlib import Path

import torch

import config
from dataset import CXRDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["image", "tabular"], required=True)
    args = parser.parse_args()
    split_path = config.SPLIT_DIR / "train.csv"
    if not split_path.is_file():
        raise SystemExit("ERROR: Split files are missing. Run python data_prep.py first.")
    dataset = CXRDataset(split_path)
    image, tabular, labels, subject_id = dataset[0]
    if args.branch == "image":
        print(f"Image check passed: shape={tuple(image.shape)}, dtype={image.dtype}, subject_id={subject_id}")
    else:
        missing_columns = [column for column in config.TABULAR_FEATURE_COLUMNS if column not in dataset.rows.columns]
        if missing_columns:
            raise SystemExit(f"ERROR: Prepared tabular split is missing columns: {missing_columns}")
        print(f"Tabular check passed: shape={tuple(tabular.shape)}, finite={bool(torch.isfinite(tabular).all())}, labels={labels.tolist()}")


if __name__ == "__main__":
    main()
