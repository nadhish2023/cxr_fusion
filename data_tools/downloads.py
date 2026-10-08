import os
from datetime import datetime
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "Dataset"
URL_FILE = DATASET_DIR / "urls_remaining.txt"
URL_PREFIX = "/2.1.0/files/"

if not os.path.exists(URL_FILE):
    print(f"Error: '{URL_FILE}' not found in the current directory.")
    exit(1)

with open(URL_FILE, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

total_urls = len(urls)
downloaded_count = 0
missing_count = 0
corrupted_or_empty = 0

for url in urls:
    if URL_PREFIX in url:
        rel_path = url.split(URL_PREFIX)[-1]
    else:
        rel_path = url.split("mimic-cxr-jpg/")[-1]

    local_path = DATASET_DIR / Path(os.path.normpath(rel_path))

    if os.path.isfile(local_path):
        if os.path.getsize(local_path) > 0:
            downloaded_count += 1
        else:
            corrupted_or_empty += 1
    else:
        missing_count += 1

percent_done = (downloaded_count / total_urls) * 100 if total_urls > 0 else 0
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("=" * 45)
print("DOWNLOAD PROGRESS SUMMARY")
print("=" * 45)
print(f"Total targets in {URL_FILE} : {total_urls}")
print(f"Successfully downloaded       : {downloaded_count} ({percent_done:.2f}%)")
print(f"Still missing                 : {missing_count}")
if corrupted_or_empty > 0:
    print(f"Corrupt / 0-byte files        : {corrupted_or_empty} (need re-download)")
print("=" * 45)
print(f"Report Generated: {current_time}")