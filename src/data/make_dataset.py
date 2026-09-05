"""Top-level, re-runnable pipeline: download -> build -> report.

Usage:
    python src/data/make_dataset.py            # full pipeline (skips cached files)
    python src/data/make_dataset.py --no-download   # rebuild from cached raw

All outputs land in data/processed/ ; raw manifest in data/raw/manifest.csv.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
import build
import report as report_mod  # noqa: F401  (import triggers module load only)


def run(no_download: bool = False) -> None:
    if not no_download:
        print("== [1/3] downloading NHANES + LMF ==")
        download.main()
    else:
        print("== [1/3] skipping download (using cached raw) ==")
    print("== [2/3] building analytic dataset ==")
    build.main()
    print("== [3/3] writing codebook + cohort flow ==")
    import importlib
    importlib.reload(report_mod)  # regenerate against fresh parquet
    report_mod.write_codebook()
    report_mod.write_flow()
    print("done. outputs in data/processed/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true",
                    help="rebuild from cached raw files without re-downloading")
    args = ap.parse_args()
    run(no_download=args.no_download)
