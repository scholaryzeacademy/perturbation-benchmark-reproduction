"""Stage 3c -- Geneformer zero-shot in silico perturbation (BUILD_PLAN.md
Sec.6 Stage 3c). Zero-shot means there's no training loop here, unlike
train_gears.py/train_scgpt.py -- this script tokenizes the real control
cells once, then for each of this project's held-out test conditions runs
Geneformer's own InSilicoPerturber (knockout/knockdown simulation) +
InSilicoPerturberStats (per-condition cosine-shift summary) and concatenates
the results into one CSV.

Output is a per-condition cosine-shift ranking, NOT a predicted expression
profile comparable to GEARS/scGPT/baselines under Stage 4's MSE@20DEG or
Pearson(Δcontrol) -- see src/stage3_models/geneformer_wrapper.py's docstring
and BUILD_PLAN.md Sec.6 Stage 3c for why, and Stage 4/Stage 6 for how this is
handled in reporting (Geneformer gets its own results table, not the same
numeric columns as the other models).

UNTESTED end-to-end as of this writing (ported directly against
ctheodoris/Geneformer's own source, not an official example -- see
CLAUDE.md's Working conventions section). Needs the perturb-bench-geneformer
env, a GPU, and a git-lfs clone of the pretrained checkpoint (see
environment-geneformer.yml's install note) -- treat its first real run as
the actual test.

Usage:
    python -m src.stage3_models.run_geneformer --dataset adamson --model-dir ./Geneformer
    python -m src.stage3_models.run_geneformer --dataset norman --model-dir ./Geneformer

--max-conditions/--max-ncells let you smoke-test on a small subset (e.g. for
validating the pipeline actually runs before committing to a full pass over
every test condition) without editing configs/training_hyperparameters.yaml.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from src.perturbation_conditions import perturbed_genes
from src.stage1_data.load_data import CONFIG_PATH, SUPPORTED_DATASETS, load_perturbation_data
from src.stage3_models.geneformer_wrapper import (
    compute_perturbation_stats,
    run_in_silico_perturbation,
    symbols_to_ensembl,
    tokenize_control_cells,
)


def _safe_prefix(condition: str) -> str:
    return condition.replace("+", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dir", required=True, help="path to the cloned Geneformer checkpoint directory (repo root for the default Geneformer-V2-316M)")
    parser.add_argument("--output-dir", default=None, help="defaults to ./data/geneformer_<dataset>")
    parser.add_argument("--max-conditions", type=int, default=None, help="only process the first N test conditions (smoke-testing)")
    parser.add_argument("--max-ncells", type=int, default=None, help="override stage3c_geneformer.max_ncells (smoke-testing)")
    args = parser.parse_args()
    output_dir = args.output_dir or f"{args.data_dir}/geneformer_{args.dataset}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    s1 = cfg["stage1_data"]
    s3c = cfg["stage3c_geneformer"]
    if args.max_ncells is not None:
        s3c = {**s3c, "max_ncells": args.max_ncells}

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

    print("=== Tokenizing control cells ===", flush=True)
    t0 = time.time()
    tokenized_path = tokenize_control_cells(
        pert_data.adata,
        output_dir=f"{output_dir}/tokenized",
        output_prefix=f"{args.dataset}_ctrl",
        model_input_size=s3c["model_input_size"],
        model_version=s3c["model_version"],
        nproc=s3c["nproc"],
    )
    print(f"=== Tokenized in {time.time() - t0:.1f}s -> {tokenized_path} ===", flush=True)

    test_conditions = [c for c in pert_data.set2conditions["test"] if c != "ctrl"]
    if args.max_conditions is not None:
        test_conditions = test_conditions[: args.max_conditions]
    print(f"=== {len(test_conditions)} test conditions, running in silico perturbation for each ===", flush=True)

    isp_dir = f"{output_dir}/isp_raw"
    stats_dir = f"{output_dir}/isp_stats"
    all_stats = []
    for i, condition in enumerate(test_conditions):
        genes = perturbed_genes(condition)
        ensembl_ids = symbols_to_ensembl(pert_data.adata, genes)
        prefix = _safe_prefix(condition)
        t0 = time.time()
        run_in_silico_perturbation(
            model_directory=args.model_dir,
            tokenized_data_path=tokenized_path,
            genes_to_perturb=ensembl_ids,
            output_directory=isp_dir,
            output_prefix=prefix,
            perturb_type=s3c["perturb_type"],
            emb_mode=s3c["emb_mode"],
            cell_emb_style=s3c["cell_emb_style"],
            forward_batch_size=s3c["forward_batch_size"],
            nproc=s3c["nproc"],
            model_version=s3c["model_version"],
            max_ncells=s3c["max_ncells"],
        )
        df = compute_perturbation_stats(
            input_directory=isp_dir,
            output_directory=stats_dir,
            output_prefix=prefix,
            genes_perturbed=ensembl_ids,
            mode=s3c["stats_mode"],
            model_version=s3c["model_version"],
        )
        df["condition"] = condition
        all_stats.append(df)
        print(
            f"[{i + 1}/{len(test_conditions)}] {condition} ({ensembl_ids}) "
            f"done in {time.time() - t0:.1f}s",
            flush=True,
        )

    combined = pd.concat(all_stats, ignore_index=True)
    combined_path = f"{output_dir}/{args.dataset}_geneformer_cosine_shift.csv"
    combined.to_csv(combined_path, index=False)
    print(f"=== Saved combined cosine-shift results to {combined_path} ===", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
