# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Stages 1–2 (data acquisition, baselines) implemented; Stages 3–6 not yet started.** `docs/BUILD_PLAN.md` is the full project specification. All three conda environments, the multi-stage `Dockerfile`, and the CI workflow exist. `src/stage3_models/` through `src/stage6_reporting/` are empty package stubs (directory + `__init__.py` only) — that's intentional scaffolding matching `docs/BUILD_PLAN.md` §7, not partial implementation. Fill each in when that stage actually starts, and update this file's commands as you go.

## Commands

```
# Run the fast unit tests (no network) — what CI's unit-tests job runs
pytest tests/ -m "not integration" -v

# Run everything including real GEARS-download/fit integration tests
pytest tests/ -v

# Stage 1 — download + split a dataset (writes into ./data)
python -m src.stage1_data.cli --dataset adamson
python -m src.stage1_data.cli --dataset norman
```

Stage 2 (`src/stage2_baselines/`) has no CLI of its own by design — `no_change.py`, `mean_baseline.py`, `ridge_baseline.py` are just importable `fit(adata, train_conditions)` / `predict(condition)` classes, exercised via `tests/test_stage2_baselines.py`. They'll get driven end-to-end once Stage 4 (metrics) or Stage 6 (reporting) exists to actually consume their predictions.

All of the above assume the `perturb-bench` conda env is active (see Environment setup below) — `gears`/`torch_geometric` aren't installed outside it, so these commands will import-error in a bare system Python.

## Environment setup

Three separate conda environments — do not merge them, see below for why.

```
# Base — Stages 1, 2, 3a, 4, 5, 6 (CPU; also what CI runs)
conda env create -f environment.yml && conda activate perturb-bench
# or, pip-only (CI / Docker):
pip install -r requirements.txt

# Stage 3b — scGPT fine-tuning (GPU)
conda env create -f environment-scgpt.yml && conda activate perturb-bench-scgpt

# Stage 3c — Geneformer zero-shot in silico perturbation (GPU)
conda env create -f environment-geneformer.yml && conda activate perturb-bench-geneformer
# then, since Geneformer isn't on PyPI:
git lfs install && git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer && pip install .
```

`environment.yml`/`requirements.txt` deliberately exclude scGPT and Geneformer — do not `pip install scgpt` into `perturb-bench`. Verified conflict: scGPT's `pyproject.toml` pins `cell-gears<0.0.3`, but the GEARS stage depends on the actual current `cell-gears` release (`0.1.2`) — the two cannot be resolved into one environment. `environment-scgpt.yml` and `environment-geneformer.yml` isolate each GPU stage per `docs/BUILD_PLAN.md` §5's explicit call to isolate GPU stages; both pin `pytorch-cuda=12.1` as a default that should be matched to whatever GPU instance Phase 0 actually provisions.

The `Dockerfile` mirrors this same three-way split as build targets (`cpu`, `scgpt`, `geneformer` — no meaningful default, always pass `--target`), on `mambaorg/micromamba` base images:

```
docker build --target cpu        -t perturb-bench:cpu .          # what CI builds/runs
docker build --target scgpt      -t perturb-bench:scgpt .
docker build --target geneformer -t perturb-bench:geneformer .
```

The `geneformer` target intentionally does not clone/install Geneformer at build time (it's gated/credentialed on Hugging Face in some versions — BUILD_PLAN.md §11 — and credentials shouldn't be baked into image layers); that's a documented post-`docker run` step instead. `.dockerignore` excludes `data/` so downloaded Perturb-seq AnnData objects never end up baked into an image layer.

## CI (`.github/workflows/ci.yml`)

Two jobs, both on `requirements.txt` (pip, not conda — matches the `cpu` Docker target):
- **`unit-tests`** — fast, no network, runs `pytest -m "not integration"`.
- **`data-smoke-test`** (depends on `unit-tests`) — runs `pytest tests/ -m integration -v` against real data: GEARS's actual download path (Stage 1) plus fitting all three baselines against the real split (Stage 2). `./data` is cached across runs by dataset name (the `data_dir` fixture in `tests/conftest.py` intentionally does *not* use pytest's `tmp_path`, so this cache actually hits). Dataset defaults to Adamson (smaller, single-gene) via the `PERTDATA_TEST_DATASET` env var, which both `test_stage1_data.py` and `test_stage2_baselines.py`'s integration tests read — trigger Norman via `workflow_dispatch` with `dataset: norman` when needed. New stages should just add their own `@pytest.mark.integration` tests; this job picks them up automatically, no `ci.yml` edits required.

## What this project is

A benchmarking reproduction project comparing single-cell perturbation-prediction models (foundation models scGPT and Geneformer, plus the specialized GEARS model) against deliberately simple baselines (no-change, mean, ridge regression) on public Perturb-seq data (Norman et al. 2019, Adamson et al. 2016).

The project reproduces two competing, unresolved 2025 findings on the same dataset and reports honestly which one holds under which evaluation metric:
1. **Ahlmann-Eltze, Huber & Anders (*Nature Methods*, 2025)** — DL/foundation models do *not* beat simple linear baselines at predicting post-perturbation gene expression.
2. **A follow-up bioRxiv preprint (Oct 2025)** — argues that conclusion is an artifact of poorly calibrated metrics, and that DL models *do* show real signal under a calibration framework (positive/negative controls, dynamic range fraction).

Implementing both sides on the same dataset — not picking a winner — is the project's actual differentiator. See `docs/BUILD_PLAN.md` §0–§2 for the full framing and Definition of Done.

## Non-goals (do not drift into these)

- Not training a new foundation model or claiming a novel modeling contribution.
- Not attempting genome-wide scale (e.g. full Replogle et al. 2022 2.5M-cell dataset) in v1.
- Not trying to definitively resolve the Ahlmann-Eltze-vs-rebuttal debate — the goal is an independent, transparent replication, not adjudication.

## Pipeline architecture

Six sequential stages, each with a distinct owner discipline (see `docs/BUILD_PLAN.md` §3, §6, §12 for full detail):

1. **Data acquisition** — Norman + Adamson via `gears`'s `PertData` class, using GEARS's own documented train/val/test splits (built around *unseen* perturbations) rather than re-splitting, to preserve comparability with published numbers.
2. **Baseline models** — no-change (predict zero effect), mean (average training-set perturbation effect), ridge regression on gene/perturbation embeddings.
3. **Specialized/foundation models**, trained in parallel against the same splits as the baselines:
   - 3a: GEARS (graph-based specialist, official training procedure/hyperparameters)
   - 3b: scGPT (fine-tuned for perturbation prediction)
   - 3c: Geneformer (zero-shot in silico perturbation — no fine-tuning)
4. **Conventional metrics** — MSE@20DEG and Pearson(Δcontrol) via `scanpy.tl.rank_genes_groups`, matching field-standard definitions exactly for comparability to published results.
5. **Metric calibration framework** — the project's key differentiator. Only implement the *core idea* of the rebuttal paper (compare each model against the gap between a no-change negative control and an "interpolated duplicate" positive control built from real replicate pairs), not a byte-for-byte reproduction of its full 13-metric/14-dataset framework. **State this scope limitation explicitly wherever calibration results are reported** — this is intentional, honest scoping per the build plan, not a shortcut to hide.
6. **Reporting** — results table across all models × baselines × both metric regimes × both datasets, plus an explicit discussion of where the reproduction agrees/disagrees with each paper and why.

## Repository structure

```
src/
├── stage1_data/             # created — load_data.py (PertData wrapper), cli.py
├── stage2_baselines/        # created — no_change.py, mean_baseline.py, ridge_baseline.py
├── stage3_models/           # stub only — gears_wrapper.py, scgpt_wrapper.py, geneformer_wrapper.py go here
├── stage4_conventional_metrics/  # stub only
├── stage5_calibration/      # stub only
└── stage6_reporting/        # stub only
notebooks/                  # not yet created — 01_data_exploration, 02_baseline_vs_models_norman,
                             # 03_baseline_vs_models_adamson, 04_calibration_analysis
configs/
└── training_hyperparameters.yaml   # created — stage1_data: + stage2_baselines: sections; add one per stage as it's built
tests/
├── conftest.py                # created — data_dir fixture (fixed ./data path, not tmp_path, so CI caching works)
├── test_stage1_data.py        # created — unit tests + one @pytest.mark.integration real-download test
└── test_stage2_baselines.py   # created — toy-data unit tests (exact arithmetic checked) + one integration test on the real split
environment.yml / requirements.txt   # created — base CPU env, Stages 1/2/3a/4/5/6
environment-scgpt.yml                # created — GPU env, Stage 3b
environment-geneformer.yml           # created — GPU env, Stage 3c
Dockerfile / .dockerignore           # created — multi-stage: cpu / scgpt / geneformer targets
.github/workflows/ci.yml             # created — unit-tests job + data-smoke-test job (runs all @pytest.mark.integration tests)
pyproject.toml                       # created — pytest config only (pythonpath, integration marker); no packaging
.gitignore                           # created
reports/final_writeup/      # not yet created
METHODS.md                  # not yet created — every hyperparameter, split, and metric definition, for independent reproducibility
LICENSES.md                 # not yet created
```

## Tech stack

- Python 3.10, Conda env (`environment.yml`, name `perturb-bench`) + separate Docker image(s) for GPU stages (scGPT's flash-attention is CUDA-version-sensitive — keep it isolated).
- `AnnData`/`scanpy` for all data structures. Core deps in `environment.yml` are pinned to match GEARS's own `requirements.txt` exactly (`numpy==1.26.4`, `pandas==2.2.2`, `scipy==1.14.1`, `scikit-learn==1.5.1`, `scanpy==1.10.2`, `networkx==3.3`, `tqdm==4.66.5`) — don't casually bump these without re-checking GEARS compatibility.
- `cell-gears` (confirmed current PyPI name/version: `0.1.2`) for GEARS. Requires PyTorch Geometric installed first (`torch_geometric`; optional accelerated ops like `torch-scatter` need wheels matched to the exact torch build).
- `scgpt` (from `github.com/bowang-lab/scGPT`) for scGPT fine-tuning — **must live in its own environment**, not `perturb-bench` (see Environment setup above for the `cell-gears` version conflict). Also pins `scanpy>=1.9.1,<2.0.0`, `scvi-tools>=0.16.0,<1.0`, `orbax<0.1.8`; needs `flash-attn` for the fast attention path (GPU/CUDA-build-specific).
- `Geneformer` (git-lfs clone from Hugging Face, `ctheodoris/Geneformer`, then `pip install .` — not on PyPI; may require gated/credentialed HF access, confirm before use) for zero-shot in silico perturbation. Requires Python >=3.10; deps (`transformers==4.46`, `peft`, `ray`, `optuna`, etc.) are permissive enough to coexist with the base env's pins, but keep it in its own environment anyway for the same GPU-isolation reason as scGPT.
- `scikit-learn` `Ridge` for the baseline.
- GitHub Actions CI smoke-tests only the baseline + GEARS pipeline on a small subset (installs from `requirements.txt`) — foundation-model fine-tuning is too heavy for CI and is run manually/on-demand.

## Working conventions specific to this project

- **Sanity-check the no-change baseline first**, before building anything else. If it doesn't land in a plausible range vs. published numbers, debug the pipeline before proceeding — it's the field's own headline empirical anchor.
- **Cross-check GEARS output against its own published Norman benchmark numbers** (e.g. ~0.176 MSE@20DEG in one commonly cited comparison) as a pipeline-correctness check before trusting scGPT/Geneformer runs.
- **Separate "doesn't match the paper" (debugging problem) from "shows a genuinely different pattern" (real finding)** — document either outcome; never silently adjust methodology until numbers look "right."
- **Run and report both Norman (combinatorial) and Adamson (single-gene)** — a conclusion that differs between the "easy" and "hard" perturbation regime is a legitimate finding to publish, not a discrepancy to paper over.
- **Every hyperparameter and random seed goes in `configs/`** — reproducibility by an independent party is a hard project requirement, not a nice-to-have, given the benchmarking-fragility literature this project is directly engaging with.
- Metric definitions must match the source papers exactly (MSE@20DEG, Pearson of delta-from-control via t-test DEG selection) — don't substitute a "close enough" metric variant.
- **Stage 2's baseline interface is the contract later stages should match**: `fit(adata: AnnData, train_conditions: list[str])` / `predict(condition: str) -> np.ndarray` (a per-gene mean expression profile, `n_genes` long). All three baselines share it — even `NoChangeBaseline`, which ignores `train_conditions` — specifically so Stage 4's evaluation loop can treat every baseline (and eventually GEARS/scGPT/Geneformer wrappers) uniformly. `RidgeBaseline` uses a multi-hot gene-identity encoding as its "perturbation embedding," a deliberate, documented simplification of the ESM-2-style embeddings BUILD_PLAN.md §5 mentions — not a hidden shortcut.

## Licensing checklist to keep in mind

Before publishing any derived outputs, GEARS/Norman/Adamson redistribution terms, scGPT and Geneformer weight licenses (Geneformer may be gated on Hugging Face), and scPerturb terms of use (if used) all need confirming — see `docs/BUILD_PLAN.md` §11. Ahlmann-Eltze et al. 2025 and the rebuttal preprint must be cited prominently since the entire project framing depends on transparently building on both.
