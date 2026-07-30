"""Stage 4 -- Conventional metrics (BUILD_PLAN.md Sec.6 Stage 4).

Field-standard metrics computed independently of any model's own internal
evaluation code, so every model (baselines, GEARS, scGPT) is scored through
the exact same code path -- this is the point of Stage 4, not a
reimplementation of what train_gears.py/train_scgpt.py already print.

MSE@20DEG: mean squared error between predicted and real mean post-
perturbation expression, restricted to the top-20 differentially expressed
genes for that condition (vs. control), selected via
`scanpy.tl.rank_genes_groups` (t-test, ranked by absolute score -- matching
GEARS's own `rank_genes_groups_by_cov` convention, `rankby_abs=True`, so our
independently-computed DE gene sets are chosen the same way GEARS's own
internal metrics choose theirs). Real GEARS runs also filter their top-20
down to a "non-dropout" gene subset (see `gears/data_utils.py`'s
`get_dropout_non_zero_genes`) before computing their own reported "Test Top
20 DE MSE" -- this module deliberately does NOT replicate that extra
filtering step, so a modest, expected numeric difference from GEARS's own
self-reported number is normal, not a correctness bug; see
run_baselines_gears.py's docstring for how this is used as a sanity check
rather than an exact-match target.

Pearson(delta control): Pearson correlation between predicted and real
expression *change from control* (not raw expression), across all genes,
per condition -- a transcriptome-wide, direction-of-effect metric,
deliberately not restricted to the top-20 DE genes (that's what MSE@20DEG is
for).

Both metrics operate on *mean* expression profiles per condition (matching
Stage 2's own predict(condition) -> np.ndarray contract: "a per-gene mean
expression profile, n_genes long"), not per-cell values -- consistent with
every model plugged into evaluate_model() below returning that same shape.

This module only needs scanpy/numpy/pandas/scipy -- no gears/scgpt/geneformer
imports -- so it is importable unmodified from any of this project's
environments (perturb-bench-gears-gpu, perturb-bench-scgpt), letting
per-model driver scripts (which do need those environment-specific packages)
share one evaluation code path.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import pearsonr


def top20_de_genes(
    adata: AnnData,
    condition: str,
    reference: str = "ctrl",
    method: str = "t-test",
    n_genes: int = 20,
) -> list[int]:
    """Top-`n_genes` differentially expressed genes for `condition` vs.
    `reference`, as integer positions into adata.var_names (matching the
    gene ordering every predict(condition) -> np.ndarray in this project
    already uses). Subsets to just the two relevant groups before calling
    rank_genes_groups -- purely a speed optimization on datasets with dozens
    of other conditions' cells that would otherwise be scanned for nothing.
    """
    sub = adata[adata.obs["condition"].isin([condition, reference])].copy()
    sub.obs["condition"] = sub.obs["condition"].astype(str).astype("category")
    sc.tl.rank_genes_groups(
        sub,
        groupby="condition",
        groups=[condition],
        reference=reference,
        method=method,
        n_genes=n_genes,
        rankby_abs=True,
        use_raw=False,
    )
    de_gene_names = sub.uns["rank_genes_groups"]["names"][condition]
    name_to_idx = {name: i for i, name in enumerate(adata.var_names)}
    return [name_to_idx[name] for name in de_gene_names]


def mse_at_20deg(pred: np.ndarray, truth_mean: np.ndarray, de_idx: list[int]) -> float:
    pred = np.asarray(pred)
    truth_mean = np.asarray(truth_mean)
    return float(np.mean((pred[de_idx] - truth_mean[de_idx]) ** 2))


def pearson_delta_control(pred: np.ndarray, truth_mean: np.ndarray, ctrl_mean: np.ndarray) -> float:
    pred_delta = np.asarray(pred) - np.asarray(ctrl_mean)
    truth_delta = np.asarray(truth_mean) - np.asarray(ctrl_mean)
    r, _ = pearsonr(pred_delta, truth_delta)
    return float(r)


def evaluate_model(
    predict_fn,
    adata: AnnData,
    test_conditions: list[str],
    model_name: str,
    reference: str = "ctrl",
    method: str = "t-test",
    n_genes: int = 20,
) -> pd.DataFrame:
    """Uniform evaluation entry point: apply MSE@20DEG and Pearson(delta
    control) to `model_name` across every condition in test_conditions,
    using `predict_fn(condition) -> np.ndarray` -- the exact Stage 2 baseline
    contract (also matched by GearsModel.predict and this project's scGPT
    wrapper). One row per condition in the returned DataFrame."""
    ctrl_mean = np.asarray(adata[adata.obs["condition"] == reference].X.mean(axis=0)).ravel()
    rows = []
    for condition in test_conditions:
        truth_cells = adata[adata.obs["condition"] == condition]
        if truth_cells.n_obs == 0:
            print(f"WARNING: 0 real cells for test condition {condition!r} -- skipping", flush=True)
            continue
        truth_mean = np.asarray(truth_cells.X.mean(axis=0)).ravel()
        pred = np.asarray(predict_fn(condition))
        de_idx = top20_de_genes(adata, condition, reference=reference, method=method, n_genes=n_genes)
        rows.append(
            {
                "model": model_name,
                "condition": condition,
                "mse_20deg": mse_at_20deg(pred, truth_mean, de_idx),
                "pearson_delta_control": pearson_delta_control(pred, truth_mean, ctrl_mean),
                "n_de_genes": len(de_idx),
            }
        )
    return pd.DataFrame(rows)
