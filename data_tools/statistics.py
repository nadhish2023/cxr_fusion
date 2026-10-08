from datetime import datetime
import os
from pathlib import Path
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent.parent / "Dataset"
CSV_FILE = DATASET_DIR / "Full_Dataset.csv"
OUTPUT_CSV = DATASET_DIR / "Available_Dataset.csv"

if not os.path.exists(CSV_FILE):
  print(f"Error: '{CSV_FILE}' not found in current directory.")
  exit(1)

print("Loading dataset and verifying downloaded files on disk...")
df = pd.read_csv(CSV_FILE)


def is_downloaded(image_path):
  clean = str(image_path).strip()
  if clean.startswith("files/"):
    clean = clean[len("files/") :]
  local_path = DATASET_DIR / Path(os.path.normpath(clean))
  return os.path.isfile(local_path) and os.path.getsize(local_path) > 0


# Filter rows to only those whose images are completely downloaded
mask = df["image_path"].apply(is_downloaded)
df_present = df[mask].copy()

total_cohort = len(df)
available_count = len(df_present)
percent_present = (
    (available_count / total_cohort) * 100 if total_cohort > 0 else 0
)

# Save/replace the available rows CSV
df_present.to_csv(OUTPUT_CSV, index=False)
print(f"✔ Saved {available_count:,} downloaded rows to '{OUTPUT_CSV}'")

print("\n" + "=" * 55)
print("CURRENT DOWNLOADED COHORT STATISTICS")
print("=" * 55)
print(f"Images Verified on Disk : {available_count:,} / {total_cohort:,}")
print(f"Cohort Completion Rate  : {percent_present:.2f}%")
print(f"Unique Patients         : {df_present['subject_id'].nunique():,}")
print(f"Unique Radiology Studies: {df_present['study_id'].nunique():,}")
print("-" * 55)

# Demographics Breakdown
print("PATIENT DEMOGRAPHICS")
if "gender" in df_present.columns:
  gender_counts = df_present["gender"].value_counts(dropna=False)
  for g, c in gender_counts.items():
    print(f"  Gender '{g}': {c:,} ({c / available_count * 100:.1f}%)")

if "anchor_age" in df_present.columns:
  print(
      f"  Age: Mean {df_present['anchor_age'].mean():.1f} ±"
      f" {df_present['anchor_age'].std():.1f} (Median:"
      f" {df_present['anchor_age'].median():.0f}, Min:"
      f" {df_present['anchor_age'].min():.0f}, Max:"
      f" {df_present['anchor_age'].max():.0f})"
  )

print("-" * 55)

# Target Pathology Distribution (CheXpert Ground Truth)
pathology_cols = ["Pneumonia", "Cardiomegaly", "Edema", "Atelectasis"]
print("PATHOLOGY TARGET LABELS (CheXpert Ground Truth)")

for col in pathology_cols:
  if col in df_present.columns:
    pos = (df_present[col] == 1.0).sum()
    neg = (df_present[col] == 0.0).sum()
    unc = (df_present[col] == -1.0).sum()
    unmentioned = df_present[col].isna().sum()

    print(f"\n  • {col}:")
    print(f"      Positive (1.0)   : {pos:,} ({pos / available_count * 100:.1f}%)")
    print(f"      Negative (0.0)   : {neg:,} ({neg / available_count * 100:.1f}%)")
    print(f"      Uncertain (-1.0) : {unc:,} ({unc / available_count * 100:.1f}%)")
    print(
        f"      Unmentioned/Null : {unmentioned:,} ({unmentioned / available_count * 100:.1f}%)"
    )

print("-" * 55)

# Clinical Vitals Summary (Derived Module)
vital_cols = ["heart_rate", "sbp", "dbp", "spo2"]
print("CLINICAL VITALS SUMMARY (Physiological Readings)")

for v in vital_cols:
  if v in df_present.columns:
    val = df_present[v].dropna()
    print(
        f"  • {v.upper():<10} -> Mean: {val.mean():.1f} | Median:"
        f" {val.median():.1f} | Min: {val.min():.1f} | Max: {val.max():.1f}"
    )

print("=" * 55)
print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 55)
