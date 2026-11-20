"""Generate the demonstration fleet and the CSV upload template.

Usage:
    python scripts/generate_demo_data.py

Writes data/sample_data.csv (M-001..M-012, 48 hourly readings, clearly labelled
as demonstration records) and data/sample_template.csv. Deterministic seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_loader


def main() -> int:
    fleet = data_loader.generate_demo_fleet()
    fleet.to_csv(config.SAMPLE_DATA_PATH, index=False)
    print(f"Wrote {config.SAMPLE_DATA_PATH}: {fleet.shape[0]} rows, "
          f"{fleet['Machine ID'].nunique()} machines")
    config.TEMPLATE_PATH.write_bytes(data_loader.sample_template_bytes())
    print(f"Wrote {config.TEMPLATE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
