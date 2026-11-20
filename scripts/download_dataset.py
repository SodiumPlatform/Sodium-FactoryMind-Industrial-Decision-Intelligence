"""Download the UCI AI4I 2020 Predictive Maintenance dataset into data/.

Usage:
    python scripts/download_dataset.py

Only needed to reproduce the committed data/ai4i2020.csv; the repository
already ships the copy and the app runs fully offline.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

URL = ("https://archive.ics.uci.edu/static/public/601/"
       "ai4i+2020+predictive+maintenance+dataset.zip")


def main() -> int:
    zip_path = Path("/tmp/ai4i.zip")
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            print("No CSV found inside the zip archive.")
            return 1
        data = zf.read(names[0])
    config.RAW_DATA_PATH.write_bytes(data)
    print(f"Wrote {config.RAW_DATA_PATH} ({len(data):,} bytes) from {names[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
