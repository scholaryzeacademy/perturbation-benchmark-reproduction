"""Stage 5 -- Metric calibration framework (BUILD_PLAN.md Sec.6 Stage 5).

Implements the *core idea* of the rebuttal paper's calibration argument --
compare each model's score against the gap between a negative control
(no-change baseline, already built in Stage 2) and a positive control (an
"interpolated duplicate" upper-bound reference) -- not a byte-for-byte
reproduction of its full 13-metric/14-dataset framework. Per BUILD_PLAN.md's
explicit honest-scope instruction for this stage, two real scope narrowings
are documented here rather than silently built around:

1. **No true biological-replicate/batch column exists in this project's real
   data.** Confirmed directly against `adata.obs` for both datasets (GEARS's
   own `perturb_processed.h5ad`): only `condition`, `cell_type`, `dose_val`,
   `control`, `condition_name` -- no batch/lane/replicate field. So the
   "interpolated duplicate" positive control here is built from *repeated
   random 50/50 splits of each condition's own real held-out cells into two
   pseudo-replicate halves* (scored both directions, half A predicting half
   B and vice versa, repeated `n_splits` times and averaged for stability),
   not genuine separate-batch replicates. This narrows BUILD_PLAN.md's own
   "held-out biological replicates" phrasing -- an honest scope decision,
   not a hidden substitution.
2. **no_change's Pearson(delta control) is mathematically NaN** (a
   zero-vector prediction has undefined correlation with anything -- see
   Stage 4's Working Conventions in CLAUDE.md). `calibrate_dataset` takes an
   explicit `pearson_negative_reference` (config: 0.0, the principled "no
   linear relationship" stand-in) instead of using no_change's own NaN.

Both `interpolated_duplicate_scores` and `calibrate_dataset` reuse
`stage4_conventional_metrics.metrics`'s `top20_de_genes`/`mse_at_20deg`/
`pearson_delta_control` directly rather than reimplementing them, so the
positive control is scored with the exact same DE-gene selection and metric
definitions every model is scored with -- a real prediction-vs-truth
comparison where "prediction" is one random half of a condition's real
cells and "truth" is the other half.

Aggregation is **pooled**, not per-condition-ratio-then-averaged: each
calibrated fraction is computed once per (dataset, model) from three
already-averaged numbers (mean MSE/Pearson across that model's own
conditions, for model/negative-control/positive-control). This avoids a
per-condition ratio blowing up when a single weak-true-effect condition's
positive control barely beats the no-change negative control for that one
condition (a near-zero denominator) -- a real, expected edge case for
perturbations with little true biological effect, not a pipeline bug.

This module has no gears/scgpt/geneformer import (same portability
convention as metrics.py) so it's importable unmodified from whichever
environment has the per-condition results to calibrate.
"""
import numpy as np
import pandas as pd

from src.stage4_conventional_metrics.metrics import mse_at_20deg, pearson_delta_control, top20_de_genes


def interpolated_duplicate_scores(
    adata,
    condition: str,
    ctrl_mean: np.ndarray,
    de_idx: list[int],
    n_splits: int,
    rng: np.random.Generator,
) -> dict:
    """One condition's positive-control scores: `n_splits` random 50/50
    splits of its real cells, each scored in both directions (half A
    predicting half B, and half B predicting half A) via the exact same
    mse_at_20deg/pearson_delta_control metrics.py uses for real models --
    2*n_splits samples total, averaged."""
    cells = adata[adata.obs["condition"] == condition]
    n_cells = cells.n_obs
    mse_samples = []
    pearson_samples = []
    for _ in range(n_splits):
        perm = rng.permutation(n_cells)
        half = n_cells // 2
        idx_a, idx_b = perm[:half], perm[half:]
        mean_a = np.asarray(cells.X[idx_a].mean(axis=0)).ravel()
        mean_b = np.asarray(cells.X[idx_b].mean(axis=0)).ravel()
        for pred, truth in ((mean_a, mean_b), (mean_b, mean_a)):
            mse_samples.append(mse_at_20deg(pred, truth, de_idx))
            pearson_samples.append(pearson_delta_control(pred, truth, ctrl_mean))
    return {
        "condition": condition,
        "n_cells": n_cells,
        "n_samples": len(mse_samples),
        "mse_20deg_positive_control": float(np.mean(mse_samples)),
        "mse_20deg_positive_control_std": float(np.std(mse_samples)),
        "pearson_delta_control_positive_control": float(np.mean(pearson_samples)),
        "pearson_delta_control_positive_control_std": float(np.std(pearson_samples)),
    }


def compute_positive_control(
    adata,
    conditions: list[str],
    reference: str = "ctrl",
    n_splits: int = 10,
    seed: int = 1,
    method: str = "t-test",
    n_genes: int = 20,
) -> pd.DataFrame:
    """Positive-control scores for every condition in `conditions`. One row
    per condition; a condition with fewer than 2 real cells can't be
    split-halved and is skipped with a printed warning rather than crashing
    the whole run (same defensive convention run_geneformer.py uses for
    per-condition failures) -- not expected to trigger on this project's
    real data (every real test condition in both datasets has >=49 cells,
    confirmed directly), but kept as a guard rather than an assumption."""
    ctrl_mean = np.asarray(adata[adata.obs["condition"] == reference].X.mean(axis=0)).ravel()
    rng = np.random.default_rng(seed)
    rows = []
    for condition in conditions:
        cells = adata[adata.obs["condition"] == condition]
        if cells.n_obs < 2:
            print(
                f"WARNING: {condition!r} has <2 real cells ({cells.n_obs}) -- cannot build a "
                f"split-half positive control, skipping",
                flush=True,
            )
            continue
        de_idx = top20_de_genes(adata, condition, reference=reference, method=method, n_genes=n_genes)
        row = interpolated_duplicate_scores(adata, condition, ctrl_mean, de_idx, n_splits, rng)
        row["n_de_genes"] = len(de_idx)
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_dataset(
    per_condition_results: pd.DataFrame,
    positive_control: pd.DataFrame,
    negative_control_model: str = "no_change",
    pearson_negative_reference: float = 0.0,
) -> pd.DataFrame:
    """Pooled calibrated-fraction score per model, from `per_condition_results`
    (columns: model, condition, mse_20deg, pearson_delta_control -- the exact
    Stage 4 evaluate_model() output shape, one row per (model, condition))
    and `positive_control` (columns: condition, mse_20deg_positive_control,
    pearson_delta_control_positive_control -- compute_positive_control()'s
    output). Each model is restricted to its own real condition set (models
    can differ, e.g. scGPT's per Stage 4's documented env/split difference)
    intersected with whatever negative/positive-control coverage exists for
    those same conditions.

    calibrated_mse_fraction = (mean_mse_negctrl - mean_mse_model) / (mean_mse_negctrl - mean_mse_posctrl)
    calibrated_pearson_fraction = (mean_pearson_model - pearson_negative_reference) / (mean_pearson_posctrl - pearson_negative_reference)

    A fraction of 1.0 means the model matches the positive-control ceiling;
    0.0 means it matches the no-change baseline (no signal captured).
    Values outside [0, 1] are possible and expected (a model can do worse
    than no-change, or the positive control's own split-half sampling noise
    can occasionally be beaten) -- not clipped, so the real number is always
    visible rather than silently hidden.

    no_change's own calibrated_mse_fraction comes out exactly 0.0 by
    construction (compared against itself) -- a pipeline-correctness sanity
    check, not a special case in the code.
    """
    negctrl = per_condition_results[per_condition_results["model"] == negative_control_model].set_index(
        "condition"
    )
    posctrl = positive_control.set_index("condition")

    rows = []
    for model, group in per_condition_results.groupby("model"):
        conditions = set(group["condition"])
        common = sorted(conditions & set(negctrl.index) & set(posctrl.index))
        missing = sorted(conditions - set(common))
        if missing:
            print(
                f"WARNING: model={model!r} missing negative/positive-control coverage for "
                f"{len(missing)} condition(s) ({missing[:5]}{'...' if len(missing) > 5 else ''}) "
                f"-- excluded from this model's calibrated score",
                flush=True,
            )

        g = group[group["condition"].isin(common)]
        neg_c = negctrl.loc[common]
        pos_c = posctrl.loc[common]

        mean_mse_model = g["mse_20deg"].mean()
        mean_mse_negctrl = neg_c["mse_20deg"].mean()
        mean_mse_posctrl = pos_c["mse_20deg_positive_control"].mean()
        mse_denom = mean_mse_negctrl - mean_mse_posctrl
        if abs(mse_denom) < 1e-12:
            print(
                f"WARNING: near-zero MSE@20DEG denominator for model={model!r} "
                f"(negctrl={mean_mse_negctrl:.6g}, posctrl={mean_mse_posctrl:.6g}) -- "
                f"calibrated_mse_fraction set to NaN rather than a huge/garbage ratio",
                flush=True,
            )
            calibrated_mse_fraction = float("nan")
        else:
            calibrated_mse_fraction = (mean_mse_negctrl - mean_mse_model) / mse_denom

        mean_pearson_model = g["pearson_delta_control"].mean()
        mean_pearson_posctrl = pos_c["pearson_delta_control_positive_control"].mean()
        pearson_denom = mean_pearson_posctrl - pearson_negative_reference
        if abs(pearson_denom) < 1e-12:
            print(
                f"WARNING: near-zero Pearson(delta control) denominator for model={model!r} "
                f"(posctrl={mean_pearson_posctrl:.6g}, negctrl_ref={pearson_negative_reference:.6g}) -- "
                f"calibrated_pearson_fraction set to NaN rather than a huge/garbage ratio",
                flush=True,
            )
            calibrated_pearson_fraction = float("nan")
        else:
            calibrated_pearson_fraction = (mean_pearson_model - pearson_negative_reference) / pearson_denom

        rows.append(
            {
                "model": model,
                "n_conditions": len(common),
                "mean_mse_model": mean_mse_model,
                "mean_mse_negctrl": mean_mse_negctrl,
                "mean_mse_posctrl": mean_mse_posctrl,
                "calibrated_mse_fraction": calibrated_mse_fraction,
                "mean_pearson_model": mean_pearson_model,
                "mean_pearson_negctrl_reference": pearson_negative_reference,
                "mean_pearson_posctrl": mean_pearson_posctrl,
                "calibrated_pearson_fraction": calibrated_pearson_fraction,
            }
        )
    return pd.DataFrame(rows)
