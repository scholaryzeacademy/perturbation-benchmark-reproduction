"""Stage 5 -- compute the interpolated-duplicate positive control and
calibrate every model's Stage 4 score against it (BUILD_PLAN.md Sec.6
Stage 5). See calibration.py's docstring for the full methodology and its
two documented scope narrowings (pseudo-replicate positive control, 0.0
Pearson negative-control reference).

Runs in the perturb-bench-gears-gpu env (or perturb-bench/base once
created) -- needs load_perturbation_data (Stage 1) for `gears`, same as
run_baselines_gears.py. Does NOT need scgpt-specific packages: this script
only reads scGPT's already-computed Stage 4 CSV as data, it never re-runs
scGPT itself.

Reads Stage 4's real per-condition results
(data/stage4_results/{dataset}_baselines_gears.csv and _scgpt.csv) rather
than recomputing every model's predictions. The one exception is no_change:
Stage 4 documented that scGPT's real test-condition set differs from
baselines/GEARS's (different cell-gears versions -> different real splits
from the same seed), so Stage 4's own no_change CSV only covers
baselines/GEARS's split and is missing the handful of conditions unique to
scGPT's split. Since NoChangeBaseline.predict() is trivial (always the real
control mean, nothing to fit), this script recomputes it fresh over the
*union* of every condition any model was actually scored on for this
dataset -- so every model's calibrated score, including scGPT's, is
compared against a negative control with full coverage rather than a
partially-missing one.

Usage:
    python -m src.stage5_calibration.run_calibration --dataset adamson
    python -m src.stage5_calibration.run_calibration --dataset norman
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from src.stage1_data.load_data import CONFIG_PATH, SUPPORTED_DATASETS, load_perturbation_data
from src.stage2_baselines.no_change import NoChangeBaseline
from src.stage4_conventional_metrics.metrics import evaluate_model
from src.stage5_calibration.calibration import calibrate_dataset, compute_positive_control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--stage4-dir", default="./data/stage4_results", help="where run_baselines_gears.py/run_scgpt.py saved their per-condition CSVs")
    parser.add_argument("--output-dir", default="./data/stage5_results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    s1 = cfg["stage1_data"]
    s5 = cfg["stage5_calibration"]

    print(f"=== Loading {args.dataset!r} (full dataset, real split config) ===", flush=True)
    t0 = time.time()
    pert_data = load_perturbation_data(
        args.dataset,
        data_dir=args.data_dir,
        split=s1["split"],
        seed=s1["seed"],
        train_gene_set_size=s1["train_gene_set_size"],
        batch_size=s1["batch_size"],
        test_batch_size=s1["test_batch_size"],
    )
    adata = pert_data.adata
    print(f"=== Data ready in {time.time() - t0:.1f}s ===", flush=True)

    stage4_dir = Path(args.stage4_dir)
    frames = []
    for suffix in ("baselines_gears", "scgpt"):
        path = stage4_dir / f"{args.dataset}_{suffix}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"WARNING: {path} not found -- skipping", flush=True)
    if not frames:
        raise RuntimeError(f"No Stage 4 results found under {stage4_dir} for dataset={args.dataset!r}")
    stage4_results = pd.concat(frames, ignore_index=True)

    all_conditions = sorted(stage4_results["condition"].unique())
    print(
        f"=== {len(all_conditions)} real condition(s) across all Stage 4 models "
        f"(union of every model's own real test split) ===",
        flush=True,
    )

    print("=== Recomputing no_change fresh over the full condition union ===", flush=True)
    no_change = NoChangeBaseline().fit(adata, train_conditions=[])
    fresh_no_change = evaluate_model(no_change.predict, adata, all_conditions, model_name="no_change")
    per_condition_results = pd.concat(
        [stage4_results[stage4_results["model"] != "no_change"], fresh_no_change],
        ignore_index=True,
    )

    print(
        f"=== Computing interpolated-duplicate positive control "
        f"({s5['n_splits']} random split(s) x2 directions per condition) ===",
        flush=True,
    )
    t0 = time.time()
    positive_control = compute_positive_control(
        adata,
        all_conditions,
        reference="ctrl",
        n_splits=s5["n_splits"],
        seed=s5["seed"],
    )
    print(f"=== Positive control ready in {time.time() - t0:.1f}s ===", flush=True)

    calibrated = calibrate_dataset(
        per_condition_results,
        positive_control,
        negative_control_model="no_change",
        pearson_negative_reference=s5["pearson_negative_control_reference"],
    )
    calibrated["dataset"] = args.dataset
    positive_control["dataset"] = args.dataset

    pc_path = output_dir / f"{args.dataset}_positive_control.csv"
    positive_control.to_csv(pc_path, index=False)
    print(f"=== Saved positive control to {pc_path} ===", flush=True)

    cal_path = output_dir / f"{args.dataset}_calibrated.csv"
    calibrated.to_csv(cal_path, index=False)
    print(f"=== Saved calibrated scores to {cal_path} ===", flush=True)
    print(calibrated.to_string(index=False), flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
