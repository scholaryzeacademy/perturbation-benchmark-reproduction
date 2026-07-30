"""Stage 4 -- combine run_baselines_gears.py's and run_scgpt.py's per-dataset
CSVs into one model x metric x dataset summary table (BUILD_PLAN.md Sec.6
Stage 4's "results table" deliverable).

Geneformer is deliberately excluded -- its zero-shot cosine-shift output
isn't a predicted expression profile, so MSE@20DEG/Pearson(delta control)
don't apply to it (see metrics.py's docstring and BUILD_PLAN.md's Stage 4
Geneformer exception). It gets its own separate table once Stage 3c's full
production run is done.

NOTE on test-condition splits: baselines/GEARS and scGPT are evaluated on
each *their own* real held-out test conditions, not an identical set --
scGPT's environment pins an older cell-gears (0.0.2) than
perturb-bench-gears-gpu's (0.1.2), and the two versions produce different
real `simulation`-split results even with the same seed=1 (confirmed: 22 vs
21 Adamson conditions, 116 vs 107 Norman conditions). This is a real,
documented environment constraint (see CLAUDE.md's Environment setup
section), not a bug -- reporting each model against its own real split
(rather than forcing an intersection-only comparison) was a deliberate
choice, so this table's per-model numbers are each individually correct but
not computed over byte-identical condition sets across every row.

Usage:
    python -m src.stage4_conventional_metrics.combine_results
"""
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("./data/stage4_results")


def main() -> None:
    frames = []
    for dataset in ("adamson", "norman"):
        for suffix in ("baselines_gears", "scgpt"):
            path = RESULTS_DIR / f"{dataset}_{suffix}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
            else:
                print(f"WARNING: {path} not found -- skipping", flush=True)

    all_results = pd.concat(frames, ignore_index=True)
    per_condition_path = RESULTS_DIR / "all_per_condition.csv"
    all_results.to_csv(per_condition_path, index=False)
    print(f"=== Saved per-condition results to {per_condition_path} ===", flush=True)

    summary = (
        all_results.groupby(["dataset", "model"])
        .agg(
            mean_mse_20deg=("mse_20deg", "mean"),
            mean_pearson_delta_control=("pearson_delta_control", "mean"),
            n_test_conditions=("condition", "count"),
        )
        .reset_index()
    )
    summary_path = RESULTS_DIR / "combined_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"=== Saved summary to {summary_path} ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
