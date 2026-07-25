import os

import numpy as np
import pytest
from anndata import AnnData

from src.stage2_baselines.mean_baseline import MeanBaseline
from src.stage2_baselines.no_change import NoChangeBaseline
from src.stage2_baselines.ridge_baseline import RidgeBaseline


def _toy_adata() -> AnnData:
    # 3 conditions x 2 cells x 4 genes, deterministic values so expected
    # baseline outputs can be hand-verified in the assertions below.
    X = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],  # ctrl
            [1.0, 1.0, 1.0, 1.0],  # ctrl
            [3.0, 1.0, 1.0, 1.0],  # GENE1+ctrl (gene 0 up by 2)
            [3.0, 1.0, 1.0, 1.0],  # GENE1+ctrl
            [1.0, 5.0, 1.0, 1.0],  # GENE2+ctrl (gene 1 up by 4)
            [1.0, 5.0, 1.0, 1.0],  # GENE2+ctrl
        ]
    )
    conditions = ["ctrl", "ctrl", "GENE1+ctrl", "GENE1+ctrl", "GENE2+ctrl", "GENE2+ctrl"]
    return AnnData(X=X, obs={"condition": conditions})


def test_no_change_predicts_control_mean():
    adata = _toy_adata()
    baseline = NoChangeBaseline().fit(adata, train_conditions=["GENE1+ctrl", "GENE2+ctrl"])
    np.testing.assert_allclose(baseline.predict("GENE1+ctrl"), [1.0, 1.0, 1.0, 1.0])


def test_mean_baseline_averages_training_deltas():
    adata = _toy_adata()
    baseline = MeanBaseline().fit(adata, train_conditions=["GENE1+ctrl", "GENE2+ctrl"])
    # deltas are [2,0,0,0] and [0,4,0,0] -> mean [1,2,0,0] -> + ctrl mean [1,1,1,1]
    np.testing.assert_allclose(baseline.predict("anything"), [2.0, 3.0, 1.0, 1.0])


def test_ridge_baseline_predicts_finite_profile_for_seen_and_unseen_genes():
    adata = _toy_adata()
    baseline = RidgeBaseline(alpha=1.0, seed=1).fit(adata, train_conditions=["GENE1+ctrl", "GENE2+ctrl"])

    seen = baseline.predict("GENE1+ctrl")
    unseen = baseline.predict("GENE3+ctrl")  # gene never in the training vocab
    assert seen.shape == (4,)
    assert unseen.shape == (4,)
    assert np.all(np.isfinite(seen))
    assert np.all(np.isfinite(unseen))


@pytest.mark.integration
def test_baselines_fit_and_predict_on_real_split(data_dir):
    """Real network call -- fits all three Stage 2 baselines against GEARS's
    actual train/test split, confirming the interface holds up against real
    data, not just the toy fixture above. Same dataset selection as
    test_stage1_data.py's integration test (default 'adamson')."""
    from src.stage1_data.load_data import load_perturbation_data

    dataset = os.environ.get("PERTDATA_TEST_DATASET", "adamson")
    pert_data = load_perturbation_data(dataset, data_dir=data_dir)
    train_conditions = pert_data.set2conditions["train"]
    test_conditions = pert_data.set2conditions["test"]
    n_genes = pert_data.adata.n_vars

    for baseline in (NoChangeBaseline(), MeanBaseline(), RidgeBaseline()):
        baseline.fit(pert_data.adata, train_conditions)
        prediction = baseline.predict(test_conditions[0])
        assert prediction.shape == (n_genes,)
        assert np.all(np.isfinite(prediction))
