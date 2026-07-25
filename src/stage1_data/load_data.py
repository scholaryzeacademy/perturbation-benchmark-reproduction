"""Stage 1 -- data acquisition via GEARS's PertData class (BUILD_PLAN.md Sec.6 Stage 1)."""
from pathlib import Path

import yaml
from gears import PertData

SUPPORTED_DATASETS = ("norman", "adamson")

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "training_hyperparameters.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["stage1_data"]


def load_perturbation_data(
    dataset: str,
    data_dir: str = "./data",
    split: str = "simulation",
    seed: int = 1,
    train_gene_set_size: float = 0.75,
    batch_size: int = 32,
    test_batch_size: int = 128,
) -> PertData:
    """Download (if needed) and load `dataset`, applying GEARS's own documented
    train/val/test split rather than re-splitting, to preserve comparability
    with published numbers (BUILD_PLAN.md Sec.4/Sec.6)."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")

    pert_data = PertData(data_dir)
    pert_data.load(data_name=dataset)
    pert_data.prepare_split(split=split, seed=seed, train_gene_set_size=train_gene_set_size)
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=test_batch_size)
    return pert_data
