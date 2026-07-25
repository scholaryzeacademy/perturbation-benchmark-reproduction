import os

import numpy as np
import pytest

from src.perturbation_conditions import perturbed_genes
from src.stage3_models.gears_wrapper import GearsModel


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        GearsModel().predict("GENE1+ctrl")


def test_save_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        GearsModel().save("/tmp/unused")


def _load_smoke_test_pert_data(dataset: str, data_dir: str, n_conditions: int = 6, n_ctrl_cells: int = 800):
    """Build a small PertData for the smoke test instead of going through
    load_perturbation_data(). GEARS's own PertData.load() unconditionally
    builds -- or, once cached, unpickles -- one torch_geometric graph per
    SINGLE CELL in the *entire* dataset (including one per control cell,
    which is ~35% of Adamson's ~68k cells on its own). That cache
    (data/<dataset>/data_pyg/cell_graphs.pkl) is what actually drives this
    test's memory footprint -- it's built during data loading and is
    completely independent of hidden_size/epochs, so the smoke test's tiny
    training config does nothing to shrink it. On Adamson it unpickles to
    several GB of RAM regardless, which OOM-killed this test locally (real
    kernel OOM-killer events, not a flaky crash) and would likely do the
    same on GitHub Actions' 7GB-RAM hosted runners.

    Reading the already-downloaded h5ad directly and routing a handful of
    perturbation conditions (plus a subsampled control pool) through
    PertData.new_data_process() makes GEARS cache a separate, proportionally
    small graph set under its own dataset folder, while still exercising the
    real load/split/dataloader/fit/predict/save/load path end-to-end."""
    import scanpy as sc
    from gears import PertData
    from gears.utils import filter_pert_in_go

    adata = sc.read_h5ad(os.path.join(data_dir, dataset, "perturb_processed.h5ad"))

    pert_data = PertData(data_dir)
    pert_data.adata = adata
    pert_data.set_pert_genes()
    in_go = adata.obs["condition"].apply(lambda c: filter_pert_in_go(c, pert_data.pert_names))
    adata = adata[in_go.values].copy()

    conditions = sorted(c for c in adata.obs["condition"].unique() if c != "ctrl")
    kept_conditions = conditions[:n_conditions]
    adata = adata[adata.obs["condition"].isin([*kept_conditions, "ctrl"]).values].copy()

    # Control cells dominate cell count and each gets its own per-cell graph
    # too (see gears/pertdata.py's create_dataset_file()) -- subsample them;
    # a pipeline smoke test doesn't need biologically faithful proportions.
    rng = np.random.default_rng(0)
    ctrl_idx = np.flatnonzero((adata.obs["condition"] == "ctrl").values)
    if len(ctrl_idx) > n_ctrl_cells:
        drop = rng.choice(ctrl_idx, size=len(ctrl_idx) - n_ctrl_cells, replace=False)
        adata = adata[~np.isin(np.arange(adata.n_obs), drop)].copy()

    # Force a deterministic, non-empty test split regardless of the tiny
    # gene pool -- prepare_split's own train_gene_set_size fraction (0.75)
    # can round to zero test genes when only a handful of conditions exist.
    test_gene = perturbed_genes(kept_conditions[-1])[0]

    pert_data.new_data_process(dataset_name=f"{dataset}_smoke_test", adata=adata)
    pert_data.prepare_split(split="simulation", seed=1, train_gene_set_size=0.75, test_perts=test_gene)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    return pert_data


@pytest.mark.integration
def test_gears_trains_and_predicts_on_real_split(data_dir):
    """Real network call (via the data_dir fixture's cached download) + a
    minimal (epochs=1, tiny hidden_size) GEARS training run on a small
    perturbation subset. BUILD_PLAN.md Sec.5 explicitly calls for GEARS to
    be smoke-tested in CI on 'a small perturbation subset', unlike the
    foundation-model stages (3b/3c), which are too heavy to run
    automatically -- see _load_smoke_test_pert_data() for why that subset
    is also load-bearing for memory, not just runtime. This is NOT the
    configuration used for reported results (see
    configs/training_hyperparameters.yaml's stage3a_gears section for
    those) -- it only confirms the training/prediction pipeline runs
    end-to-end without crashing."""
    dataset = os.environ.get("PERTDATA_TEST_DATASET", "adamson")
    pert_data = _load_smoke_test_pert_data(dataset, data_dir)
    test_conditions = pert_data.set2conditions["test"]

    model = GearsModel(device="cpu", hidden_size=8, epochs=1)
    model.fit(pert_data)

    # GEARS's own predict() (gears/utils.py's create_cell_graph_dataset_for_
    # prediction) draws 300 control cells via unseeded np.random.randint and
    # averages the model output over them -- reseed immediately before each
    # predict() call below so the two calls draw the same control cells and
    # are actually comparable; otherwise this is comparing two independent
    # random samples, not testing the checkpoint round trip.
    np.random.seed(0)
    prediction = model.predict(test_conditions[0])
    assert prediction.shape == (pert_data.adata.n_vars,)
    assert np.all(np.isfinite(prediction))

    # Exercise the save()/load() round trip -- a fit-and-predict-only test
    # would never catch a broken checkpoint path, and BUILD_PLAN.md Sec.8
    # flags GEARS training as expensive enough that reloading instead of
    # retraining is a real, load-bearing part of this wrapper.
    checkpoint_path = os.path.join(data_dir, "gears_smoke_test_ckpt")
    model.save(checkpoint_path)

    reloaded = GearsModel(device="cpu")
    reloaded.load(pert_data, checkpoint_path)
    np.random.seed(0)
    reloaded_prediction = reloaded.predict(test_conditions[0])
    np.testing.assert_allclose(reloaded_prediction, prediction, rtol=1e-5)
