# Perturbation Benchmark Reproduction

[![CI](https://github.com/scholaryzeacademy/perturbation-benchmark-reproduction/actions/workflows/ci.yml/badge.svg)](https://github.com/scholaryzeacademy/perturbation-benchmark-reproduction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An independent, transparent reproduction of an unresolved 2025 single-cell perturbation-prediction benchmarking debate: Ahlmann-Eltze, Huber & Anders (*Nature Methods*, 2025) argue deep-learning and foundation models (GEARS, scGPT, Geneformer) don't yet outperform simple linear baselines; a follow-up bioRxiv preprint (Oct. 2025) argues that conclusion is an artifact of poorly calibrated metrics. This project reproduces both sides on the same public Perturb-seq datasets — Norman et al. 2019 (combinatorial) and Adamson et al. 2016 (single-gene) — under both conventional metrics and a scoped-down calibration framework, and reports honestly which finding holds under which regime, rather than picking a winner.

**Read the results**: `reports/final_writeup/technical_writeup.md` (full write-up + discussion) or `reports/final_writeup/one_page_summary.pdf` (one-page summary).

## Repository map

- `docs/BUILD_PLAN.md` — full project specification (architecture, stage-by-stage detail, definition of done).
- `METHODS.md` — every hyperparameter, random seed, data split, and metric definition, for independent reproducibility.
- `LICENSES.md` — licensing and attribution for this project's own code plus every upstream dependency, model, and dataset.
- `src/` — one package per pipeline stage (`stage1_data` through `stage6_reporting`); see `CLAUDE.md` for the full command reference and per-stage working notes.
- `configs/training_hyperparameters.yaml` — every hyperparameter and seed, machine-readable.
- `tests/` — fast unit tests (`pytest -m "not integration"`, what CI runs) plus real-data integration tests (`pytest -m integration`).
- `reports/final_writeup/` — the technical write-up and one-page summary described above.

**Not yet built**: the four exploratory/results notebooks (`notebooks/01_data_exploration.ipynb` etc.) named in `docs/BUILD_PLAN.md`'s repository-structure diagram — everything they'd cover is already in the write-up and the per-stage result CSVs under `data/`, but the notebooks themselves don't exist yet.

## Getting started

Six sequential stages, each with its own conda environment where GPU/dependency isolation matters (baselines/GEARS, scGPT, and Geneformer pin incompatible library versions — see `CLAUDE.md`'s Environment setup section for why). Quickstart:

```
conda env create -f environment.yml && conda activate perturb-bench
python -m src.stage1_data.cli --dataset adamson
pytest tests/ -m "not integration" -v
```

Full commands for every stage (GPU training runs, Docker targets, the calibration framework, report generation) are in `CLAUDE.md`.

## License

MIT — see `LICENSE`. Upstream dependencies, models, and datasets keep their own licenses; see `LICENSES.md` for the full breakdown, including two items still open for legal review before any redistribution of derived outputs.
