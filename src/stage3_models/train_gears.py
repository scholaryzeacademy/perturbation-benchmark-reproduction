"""Real Stage 3a GEARS training run (BUILD_PLAN.md Sec.6): trains with
GEARS's own documented default hyperparameters (configs/training_hyperparameters.yaml's
stage3a_gears section, excluding its smoke_test sub-block) on the full
Norman or Adamson training split, saves a checkpoint, and reports
predictions on every held-out test condition. This is the actual benchmark
run -- distinct from tests/test_stage3a_gears.py's tiny smoke-test config,
which only confirms the pipeline runs end-to-end without crashing.

Runs on CPU or GPU (auto-detects; pass --device to force one). On a laptop
CPU this is slow enough to be impractical (~15-17 hours measured for
Adamson's 20 epochs on an i7-10510U) -- see environment-gears-gpu.yml and
scripts/run_gears_gpu.sh for running this on a CUDA GPU instead.

Usage:
    python -m src.stage3_models.train_gears --dataset adamson
    python -m src.stage3_models.train_gears --dataset norman
"""
import argparse
import time

import numpy as np
import yaml

from src.stage1_data.load_data import CONFIG_PATH, SUPPORTED_DATASETS, load_perturbation_data
from src.stage3_models.gears_wrapper import GearsModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detects if omitted")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    s1 = cfg["stage1_data"]
    s3a = {k: v for k, v in cfg["stage3a_gears"].items() if k != "smoke_test"}

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
    print(f"=== Data ready in {time.time() - t0:.1f}s ===", flush=True)

    print(f"=== Training GEARS with real hyperparameters: {s3a} ===", flush=True)
    t0 = time.time()
    model = GearsModel(device=args.device, **s3a)
    model.fit(pert_data)
    print(f"=== Training done in {time.time() - t0:.1f}s (device={model.device}) ===", flush=True)

    ckpt_path = f"{args.data_dir}/gears_{args.dataset}_real_ckpt"
    model.save(ckpt_path)
    print(f"=== Saved checkpoint to {ckpt_path} ===", flush=True)

    test_conditions = pert_data.set2conditions["test"]
    print(f"=== {len(test_conditions)} test conditions, predicting all ===", flush=True)
    preds = model.predict_many(test_conditions)
    for cond in test_conditions:
        p = preds[cond]
        print(f"{cond}: shape={p.shape} mean={p.mean():.4f} finite={np.all(np.isfinite(p))}", flush=True)

    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
