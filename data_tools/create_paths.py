import os
from pathlib import Path
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent.parent / "Dataset"
CSV_FILE = DATASET_DIR / "Full_Dataset.csv"
OUTPUT_FILE = DATASET_DIR / "urls_remaining.txt"
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/"

df = pd.read_csv(CSV_FILE)

missing_urls = []
already_present = 0

for _, row in df.iterrows():
    raw_path = str(row["image_path"]).strip()
    clean_path = raw_path[len("files/"):] if raw_path.startswith("files/") else raw_path
    
    local_file = DATASET_DIR / Path(os.path.normpath(clean_path))
    
    if os.path.isfile(local_file):
        already_present += 1
    else:
        missing_urls.append(BASE_URL + clean_path.replace("\\", "/"))

print(f"Total rows: {len(df)}")
print(f"Already downloaded locally: {already_present}")
print(f"To download: {len(missing_urls)}")

with open(OUTPUT_FILE, "w") as f:
    for url in missing_urls:
        f.write(url + "\n")

print(f"Done creating {OUTPUT_FILE}")