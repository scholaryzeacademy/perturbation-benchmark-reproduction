"""Stage 6 -- generate the markdown result tables the technical write-up
embeds (BUILD_PLAN.md Sec.6 Stage 6 / Sec.13 deliverables checklist).

Reads Stage 4's conventional-metrics summary, Stage 5's calibrated summary,
and Stage 3c's Geneformer cosine-shift output -- all already computed real
CSVs, no new modeling or metric computation here. This script exists so the
write-up's numbers are generated from the source CSVs, not hand-transcribed
(the same convention Stage 4/5's combine_*.py scripts already follow, and a
real error class this project has already hit once: CLAUDE.md's own Stage 4
narrative had a wrong claim about which model "wins" Norman until it was
caught by re-deriving from the CSV directly during Stage 5's work).

Usage:
    python -m src.stage6_reporting.build_report_tables
"""
from pathlib import Path

import pandas as pd

STAGE4_DIR = Path("./data/stage4_results")
STAGE5_DIR = Path("./data/stage5_results")
GENEFORMER_DIR = Path("./data")
OUTPUT_DIR = Path("./data/stage6_results")


def build_main_results_table() -> str:
    conventional = pd.read_csv(STAGE4_DIR / "combined_summary.csv")
    calibrated = pd.read_csv(STAGE5_DIR / "calibrated_summary.csv")

    merged = conventional.merge(
        calibrated[["dataset", "model", "calibrated_mse_fraction", "calibrated_pearson_fraction"]],
        on=["dataset", "model"],
        how="left",
    )

    lines = []
    for dataset in sorted(merged["dataset"].unique()):
        lines.append(f"### {dataset}\n")
        lines.append(
            "| model | mean MSE@20DEG | mean Pearson(Δcontrol) | calibrated MSE fraction | "
            "calibrated Pearson fraction | n |"
        )
        lines.append("|---|---|---|---|---|---|")
        sub = merged[merged["dataset"] == dataset].sort_values("model")
        for _, row in sub.iterrows():
            pearson = "NaN" if pd.isna(row["mean_pearson_delta_control"]) else f"{row['mean_pearson_delta_control']:.4f}"
            cal_pearson = "NaN" if pd.isna(row["calibrated_pearson_fraction"]) else f"{row['calibrated_pearson_fraction']:.4f}"
            lines.append(
                f"| {row['model']} | {row['mean_mse_20deg']:.4f} | {pearson} | "
                f"{row['calibrated_mse_fraction']:.4f} | {cal_pearson} | {int(row['n_test_conditions'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_geneformer_summary_table(top_n: int = 10) -> str:
    lines = []
    for dataset in ("adamson", "norman"):
        path = GENEFORMER_DIR / f"geneformer_{dataset}" / f"{dataset}_geneformer_cosine_shift.csv"
        if not path.exists():
            print(f"WARNING: {path} not found -- skipping", flush=True)
            continue
        df = pd.read_csv(path)
        per_condition = (
            df.groupby("condition")["Cosine_sim"]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": "mean_cosine_sim", "std": "std_cosine_sim", "count": "n_cells"})
            .sort_values("mean_cosine_sim")
        )

        lines.append(f"### {dataset} ({len(per_condition)} conditions)\n")
        lines.append(
            "Ranked ascending by mean cosine similarity (lower similarity = bigger "
            "Geneformer-predicted transcriptional shift from the perturbation).\n"
        )
        lines.append("**Top 10 largest predicted shift:**\n")
        lines.append("| condition | mean Cosine_sim | std | n cells |")
        lines.append("|---|---|---|---|")
        for condition, row in per_condition.head(top_n).iterrows():
            lines.append(f"| {condition} | {row['mean_cosine_sim']:.6f} | {row['std_cosine_sim']:.6f} | {int(row['n_cells'])} |")
        lines.append("")
        lines.append("**Bottom 10 smallest predicted shift:**\n")
        lines.append("| condition | mean Cosine_sim | std | n cells |")
        lines.append("|---|---|---|---|")
        for condition, row in per_condition.tail(top_n).iterrows():
            lines.append(f"| {condition} | {row['mean_cosine_sim']:.6f} | {row['std_cosine_sim']:.6f} | {int(row['n_cells'])} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    main_results = build_main_results_table()
    main_path = OUTPUT_DIR / "main_results_table.md"
    main_path.write_text(main_results)
    print(f"=== Saved {main_path} ===", flush=True)

    geneformer_summary = build_geneformer_summary_table()
    geneformer_path = OUTPUT_DIR / "geneformer_summary_table.md"
    geneformer_path.write_text(geneformer_summary)
    print(f"=== Saved {geneformer_path} ===", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
