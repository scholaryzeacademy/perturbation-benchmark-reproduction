import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from src.stage4_conventional_metrics.metrics import top20_de_genes
from src.stage5_calibration.calibration import calibrate_dataset, compute_positive_control, interpolated_duplicate_scores


def _toy_adata(n_cells_per_condition: int = 20, n_genes: int = 5, seed: int = 0) -> AnnData:
    # Small per-cell noise (not exact duplicate rows) so the split-half
    # positive control and scanpy's t-test DE selection both see real
    # within-condition variance, matching real Perturb-seq data's shape --
    # deterministic (fixed seed) so test assertions are reproducible.
    rng = np.random.default_rng(seed)
    rows, conditions = [], []
    for _ in range(n_cells_per_condition):
        rows.append(rng.normal(loc=1.0, scale=0.05, size=n_genes))
        conditions.append("ctrl")
    for _ in range(n_cells_per_condition):
        vec = rng.normal(loc=1.0, scale=0.05, size=n_genes)
        vec[0] += 2.0  # gene 0 up
        rows.append(vec)
        conditions.append("GENE1+ctrl")
    for _ in range(n_cells_per_condition):
        vec = rng.normal(loc=1.0, scale=0.05, size=n_genes)
        vec[1] += 3.0  # gene 1 up
        rows.append(vec)
        conditions.append("GENE2+ctrl")
    return AnnData(X=np.vstack(rows), obs={"condition": conditions})


def test_interpolated_duplicate_scores_returns_finite_values():
    adata = _toy_adata()
    ctrl_mean = np.asarray(adata[adata.obs["condition"] == "ctrl"].X.mean(axis=0)).ravel()
    de_idx = top20_de_genes(adata, "GENE1+ctrl", reference="ctrl", n_genes=3)
    rng = np.random.default_rng(1)

    result = interpolated_duplicate_scores(adata, "GENE1+ctrl", ctrl_mean, de_idx, n_splits=5, rng=rng)

    assert result["n_cells"] == 20
    assert result["n_samples"] == 10  # 5 splits x 2 directions
    assert np.isfinite(result["mse_20deg_positive_control"])
    assert result["mse_20deg_positive_control"] >= 0
    assert np.isfinite(result["pearson_delta_control_positive_control"])


def test_compute_positive_control_covers_every_requested_condition():
    adata = _toy_adata()
    pc = compute_positive_control(
        adata, ["GENE1+ctrl", "GENE2+ctrl"], reference="ctrl", n_splits=5, seed=1, n_genes=3
    )
    assert set(pc["condition"]) == {"GENE1+ctrl", "GENE2+ctrl"}
    assert (pc["n_cells"] == 20).all()
    assert pc["mse_20deg_positive_control"].apply(np.isfinite).all()
    assert pc["pearson_delta_control_positive_control"].apply(np.isfinite).all()


def test_calibrate_dataset_pooled_arithmetic_matches_hand_computed_values():
    per_condition_results = pd.DataFrame(
        [
            {"model": "no_change", "condition": "c1", "mse_20deg": 1.0, "pearson_delta_control": np.nan},
            {"model": "no_change", "condition": "c2", "mse_20deg": 1.0, "pearson_delta_control": np.nan},
            {"model": "modelA", "condition": "c1", "mse_20deg": 0.5, "pearson_delta_control": 0.6},
            {"model": "modelA", "condition": "c2", "mse_20deg": 0.3, "pearson_delta_control": 0.8},
        ]
    )
    positive_control = pd.DataFrame(
        [
            {"condition": "c1", "mse_20deg_positive_control": 0.2, "pearson_delta_control_positive_control": 0.9},
            {"condition": "c2", "mse_20deg_positive_control": 0.0, "pearson_delta_control_positive_control": 1.0},
        ]
    )

    result = calibrate_dataset(
        per_condition_results, positive_control, negative_control_model="no_change", pearson_negative_reference=0.0
    ).set_index("model")

    # no_change compared against itself -> exactly 0.0, a pipeline sanity check.
    assert result.loc["no_change", "calibrated_mse_fraction"] == 0.0

    # mean_mse_model=(0.5+0.3)/2=0.4, mean_mse_negctrl=1.0, mean_mse_posctrl=(0.2+0.0)/2=0.1
    # calibrated = (1.0-0.4)/(1.0-0.1) = 0.6/0.9
    assert result.loc["modelA", "calibrated_mse_fraction"] == pytest.approx(0.6 / 0.9)

    # mean_pearson_model=(0.6+0.8)/2=0.7, mean_pearson_posctrl=(0.9+1.0)/2=0.95
    # calibrated = (0.7-0.0)/(0.95-0.0)
    assert result.loc["modelA", "calibrated_pearson_fraction"] == pytest.approx(0.7 / 0.95)


def test_calibrate_dataset_zero_denominator_returns_nan_not_inf():
    per_condition_results = pd.DataFrame(
        [
            {"model": "no_change", "condition": "c1", "mse_20deg": 1.0, "pearson_delta_control": np.nan},
            {"model": "modelA", "condition": "c1", "mse_20deg": 0.5, "pearson_delta_control": 0.5},
        ]
    )
    positive_control = pd.DataFrame(
        [{"condition": "c1", "mse_20deg_positive_control": 1.0, "pearson_delta_control_positive_control": 0.0}]
    )

    result = calibrate_dataset(per_condition_results, positive_control).set_index("model")

    assert np.isnan(result.loc["modelA", "calibrated_mse_fraction"])
    assert np.isnan(result.loc["modelA", "calibrated_pearson_fraction"])
    assert not np.isinf(result.loc["modelA", "calibrated_mse_fraction"])
