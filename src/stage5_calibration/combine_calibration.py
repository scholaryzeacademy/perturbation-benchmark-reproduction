"""Stage 5 -- combine run_calibration.py's per-dataset calibrated CSVs into
one model x dataset summary table (BUILD_PLAN.md Sec.6 Stage 5's results
deliverable), mirroring Stage 4's combine_results.py.

Usage:
    python -m src.stage5_calibration.combine_calibration
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("./data/stage5_results")


def main() -> None:
    frames = []
    for dataset in ("adamson", "norman"):
        path = RESULTS_DIR / f"{dataset}_calibrated.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"WARNING: {path} not found -- skipping", flush=True)

    if not frames:
        raise RuntimeError(f"No calibrated results found under {RESULTS_DIR}")

    summary = pd.concat(frames, ignore_index=True)
    summary_path = RESULTS_DIR / "calibrated_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"=== Saved calibrated summary to {summary_path} ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
