"""Stage 4 -- evaluate the no-change/mean/ridge baselines and GEARS on real
held-out test conditions, via the uniform metrics.evaluate_model() contract
(BUILD_PLAN.md Sec.6 Stage 4).

Runs in the perturb-bench-gears-gpu env (or perturb-bench/base env, once
created) -- NOT perturb-bench-scgpt, since cell-gears versions conflict
between the two (see CLAUDE.md's Environment setup section). scGPT is
evaluated by the separate run_scgpt.py, in its own environment; metrics.py
itself has no gears/scgpt import so both driver scripts share the exact
same MSE@20DEG/Pearson(delta control) code, just from different envs.

Baselines are fit fresh (fast, CPU-only, no checkpoint needed). GEARS is
loaded from the real checkpoint train_gears.py already saved
(data/gears_{dataset}_real_ckpt) -- no retraining, reusing the exact model
whose real training run CLAUDE.md documents.

Sanity check per BUILD_PLAN.md Sec.9: compare this script's own
independently-computed GEARS MSE@20DEG against train_gears.py's own
self-reported "Test Top 20 DE MSE" (0.2364 Adamson, 0.1683 Norman, per
CLAUDE.md's Working Conventions). These are NOT expected to match exactly --
GEARS's own number additionally filters its top-20 DE genes down to a
"non-dropout" subset (gears/data_utils.py's get_dropout_non_zero_genes)
before computing MSE, which this project's independent metrics.py
deliberately does not replicate (see metrics.py's docstring) -- being in the
same ballpark is the actual bar for "pipeline is probably correct", not
exact agreement.

Usage:
    python -m src.stage4_conventional_metrics.run_baselines_gears --dataset adamson
    python -m src.stage4_conventional_metrics.run_baselines_gears --dataset norman
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from src.stage1_data.load_data import CONFIG_PATH, SUPPORTED_DATASETS, load_perturbation_data
from src.stage2_baselines.mean_baseline import MeanBaseline
from src.stage2_baselines.no_change import NoChangeBaseline
from src.stage2_baselines.ridge_baseline import RidgeBaseline
from src.stage3_models.gears_wrapper import GearsModel
from src.stage4_conventional_metrics.metrics import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--gears-ckpt", default=None, help="defaults to {data-dir}/gears_{dataset}_real_ckpt (train_gears.py's own save location)")
    parser.add_argument("--output-dir", default="./data/stage4_results")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gears_ckpt = args.gears_ckpt or f"{args.data_dir}/gears_{args.dataset}_real_ckpt"

    with open(CONFIG_PATH) as f:
        s1 = yaml.safe_load(f)["stage1_data"]

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

    adata = pert_data.adata
    train_conditions = pert_data.set2conditions["train"]
    test_conditions = [c for c in pert_data.set2conditions["test"] if c != "ctrl"]
    print(f"=== {len(test_conditions)} test conditions ===", flush=True)

    all_results = []

    print("=== Fitting Stage 2 baselines ===", flush=True)
    for name, baseline in [
        ("no_change", NoChangeBaseline()),
        ("mean", MeanBaseline()),
        ("ridge", RidgeBaseline()),
    ]:
        t0 = time.time()
        baseline.fit(adata, train_conditions)
        df = evaluate_model(baseline.predict, adata, test_conditions, model_name=name)
        all_results.append(df)
        print(
            f"[{name}] mean MSE@20DEG={df['mse_20deg'].mean():.4f} "
            f"mean Pearson(dctrl)={df['pearson_delta_control'].mean():.4f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    print(f"=== Loading GEARS checkpoint from {gears_ckpt} ===", flush=True)
    t0 = time.time()
    gears_model = GearsModel().load(pert_data, gears_ckpt)
    df = evaluate_model(gears_model.predict, adata, test_conditions, model_name="gears")
    all_results.append(df)
    print(
        f"[gears] mean MSE@20DEG={df['mse_20deg'].mean():.4f} "
        f"mean Pearson(dctrl)={df['pearson_delta_control'].mean():.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    combined = pd.concat(all_results, ignore_index=True)
    combined["dataset"] = args.dataset
    out_path = output_dir / f"{args.dataset}_baselines_gears.csv"
    combined.to_csv(out_path, index=False)
    print(f"=== Saved {out_path} ===", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
