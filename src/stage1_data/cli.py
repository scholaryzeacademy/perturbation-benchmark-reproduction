"""Command-line entry point for Stage 1 data acquisition.

Usage:
    python -m src.stage1_data.cli --dataset adamson
    python -m src.stage1_data.cli --dataset norman --data-dir ./data
"""
import argparse

from src.stage1_data.load_data import SUPPORTED_DATASETS, load_config, load_perturbation_data


def main() -> None:
    defaults = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-dir", default=defaults["data_dir"])
    parser.add_argument("--split", default=defaults["split"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--train-gene-set-size", type=float, default=defaults["train_gene_set_size"])
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--test-batch-size", type=int, default=defaults["test_batch_size"])
    args = parser.parse_args()

    pert_data = load_perturbation_data(
        dataset=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        seed=args.seed,
        train_gene_set_size=args.train_gene_set_size,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )
    n_perts = pert_data.adata.obs["condition"].nunique()
    print(
        f"Loaded {args.dataset!r}: {pert_data.adata.n_obs} cells, "
        f"{n_perts} conditions, split={args.split!r} (seed={args.seed})"
    )


if __name__ == "__main__":
    main()
