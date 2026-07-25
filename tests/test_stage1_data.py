import pytest

from src.stage1_data.load_data import SUPPORTED_DATASETS, load_config, load_perturbation_data


def test_load_config_lists_all_supported_datasets():
    config = load_config()
    assert set(config["datasets"]) == set(SUPPORTED_DATASETS)


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_perturbation_data("not_a_real_dataset")


@pytest.mark.integration
def test_adamson_loads_and_splits(tmp_path):
    """Real network call against GEARS's download path -- the exact failure
    point BUILD_PLAN.md Sec.4 flags as needing to be checked first, since a
    broken download blocks everything downstream. Excluded from the fast
    unit-test job; CI's data-smoke-test job runs it explicitly."""
    pert_data = load_perturbation_data("adamson", data_dir=str(tmp_path))
    assert pert_data.adata.n_obs > 0
    assert "condition" in pert_data.adata.obs.columns
