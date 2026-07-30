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
    parser.add_argument("--conditions", default=None, help="comma-separated exact condition strings (e.g. 'CBL+PTPN9,DUSP9+ctrl') to process instead of a --max-conditions prefix -- for hand-picked smoke-test subsets spanning specific test subgroups (e.g. Norman's combo_seen0/1/2)")
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
    if args.conditions is not None:
        requested = [c.strip() for c in args.conditions.split(",")]
        missing = [c for c in requested if c not in test_conditions]
        if missing:
            raise ValueError(f"--conditions not in the real test split: {missing}")
        test_conditions = requested
    elif args.max_conditions is not None:
        test_conditions = test_conditions[: args.max_conditions]
    print(f"=== {len(test_conditions)} test conditions, running in silico perturbation for each ===", flush=True)

    all_stats = []
    succeeded_this_run = set()
    skipped = []
    for i, condition in enumerate(test_conditions):
        genes = perturbed_genes(condition)
        ensembl_ids = symbols_to_ensembl(pert_data.adata, genes)
        prefix = _safe_prefix(condition)
        # Per-condition subdirectories, not one shared isp_raw/isp_stats for
        # the whole run: InSilicoPerturberStats.get_stats -> read_dictionaries
        # scans EVERY "*_raw.pickle" file in input_data_directory by suffix
        # match alone, not filtered by output_prefix (confirmed directly
        # against in_silico_perturber_stats.py's read_dictionaries source).
        # A shared directory across conditions means each condition's stats
        # call re-reads every prior condition's pickle too -- harmless (if
        # wasteful) when every condition is single-gene, since the dict keys
        # are all plain ints and the later Ensembl_ID filter still isolates
        # the right row. But mixing combo (tuple-keyed) and single-gene
        # (int-keyed) conditions in the same run -- Norman's real case --
        # makes get_gene_list's gene_list.sort() crash with
        # `TypeError: '<' not supported between instances of 'int' and
        # 'tuple'`, confirmed by a real crash processing a single-gene
        # condition after 3 combo conditions had already written pickles to
        # the same shared directory.
        isp_dir = f"{output_dir}/isp_raw/{prefix}"
        stats_dir = f"{output_dir}/isp_stats/{prefix}"
        t0 = time.time()
        # Broad except is deliberate: a real full-dataset run over many
        # conditions can hit per-condition failures this project doesn't
        # control -- e.g. some gene pairs have zero control cells expressing
        # both genes simultaneously ("delete" perturbation needs the gene(s)
        # already tokenized in a cell to simulate deleting them), which
        # raises a bare `RuntimeError` deep in geneformer's own
        # perturber_utils.py with no more specific exception type to catch.
        # One bad condition failing shouldn't lose every other condition's
        # already-computed results (or abort a run that's hours into a large
        # Norman pass) -- skip it, log why, and keep going. Ctrl-C/SystemExit
        # still propagate normally since they're not Exception subclasses.
        try:
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
        except Exception as e:
            print(
                f"[{i + 1}/{len(test_conditions)}] {condition} ({ensembl_ids}) "
                f"FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {e}",
                flush=True,
            )
            skipped.append(
                {"condition": condition, "ensembl_ids": ensembl_ids, "error": f"{type(e).__name__}: {e}"}
            )
            continue

        df["condition"] = condition
        all_stats.append(df)
        succeeded_this_run.add(condition)
        print(
            f"[{i + 1}/{len(test_conditions)}] {condition} ({ensembl_ids}) "
            f"done in {time.time() - t0:.1f}s",
            flush=True,
        )

    # Merge with any pre-existing combined/skipped CSVs from a prior run
    # rather than overwriting them outright -- a common real usage pattern
    # is re-running just a --conditions subset (e.g. retrying conditions
    # that failed on GPU OOM last time), and a previous version of this
    # script clobbered a prior run's full results down to just the subset
    # just re-run (confirmed by hitting it for real: a 12-condition Adamson
    # combined CSV was reduced to 9 rows' worth of conditions after
    # re-running the other 9 -- recovered only because per-condition
    # isp_stats/<prefix>/ subdirectories from the earlier run were still on
    # disk and could be manually reconstructed). Only rows for conditions
    # this run actually produced fresh *successful* output for are replaced
    # -- a condition that was already successful in a prior run and happens
    # to fail on a re-run keeps its old good data rather than losing it.
    test_conditions_set = set(test_conditions)

    skipped_path = f"{output_dir}/{args.dataset}_geneformer_skipped_conditions.csv"
    skipped_df = pd.DataFrame(skipped, columns=["condition", "ensembl_ids", "error"])
    if Path(skipped_path).exists():
        existing_skipped = pd.read_csv(skipped_path)
        existing_skipped = existing_skipped[~existing_skipped["condition"].isin(test_conditions_set)]
        skipped_df = pd.concat([existing_skipped, skipped_df], ignore_index=True)

    if not skipped_df.empty:
        skipped_df.to_csv(skipped_path, index=False)
        print(
            f"=== {len(skipped)}/{len(test_conditions)} condition(s) skipped this run after a "
            f"per-condition failure -- {len(skipped_df)} total skipped condition(s) now recorded "
            f"in {skipped_path} ===",
            flush=True,
        )
    elif Path(skipped_path).exists():
        # Every condition that was previously skipped and is in scope of
        # this run now succeeded -- the file would otherwise be stale.
        Path(skipped_path).unlink()

    if not all_stats:
        raise RuntimeError(
            f"All {len(test_conditions)} condition(s) failed -- see the skipped-conditions log above."
        )

    new_combined = pd.concat(all_stats, ignore_index=True)
    combined_path = f"{output_dir}/{args.dataset}_geneformer_cosine_shift.csv"
    if Path(combined_path).exists():
        existing_combined = pd.read_csv(combined_path)
        existing_combined = existing_combined[~existing_combined["condition"].isin(succeeded_this_run)]
        combined = pd.concat([existing_combined, new_combined], ignore_index=True)
    else:
        combined = new_combined
    combined.to_csv(combined_path, index=False)
    print(
        f"=== Saved combined cosine-shift results to {combined_path} "
        f"({combined['condition'].nunique()} total condition(s)) ===",
        flush=True,
    )
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
