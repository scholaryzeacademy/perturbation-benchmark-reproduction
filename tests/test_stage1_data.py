import os

import pytest

from src.stage1_data.load_data import SUPPORTED_DATASETS, load_config, load_perturbation_data


def test_load_config_lists_all_supported_datasets():
    config = load_config()
    assert set(config["datasets"]) == set(SUPPORTED_DATASETS)


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_perturbation_data("not_a_real_dataset")


@pytest.mark.integration
def test_dataset_loads_and_splits(data_dir):
    """Real network call against GEARS's download path -- the exact failure
    point BUILD_PLAN.md Sec.4 flags as needing to be checked first, since a
    broken download blocks everything downstream. Excluded from the fast
    unit-test job; CI's data-smoke-test job runs it explicitly.

    Dataset is env-var-selectable (default 'adamson', the smaller one) so CI
    can smoke-test Norman on demand via workflow_dispatch without editing
    this test -- see .github/workflows/ci.yml."""
    dataset = os.environ.get("PERTDATA_TEST_DATASET", "adamson")
    pert_data = load_perturbation_data(dataset, data_dir=data_dir)
    assert pert_data.adata.n_obs > 0
    assert "condition" in pert_data.adata.obs.columns
