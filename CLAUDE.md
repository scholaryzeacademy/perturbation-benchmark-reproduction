# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Stages 1–3a/3b (data acquisition, baselines, GEARS, scGPT) implemented and verified with real runs; Stage 3c (Geneformer) validated on small real subsets of both Adamson and Norman (RTX 4090) with six real bugs found and fixed, then run for real on both datasets' full condition sets (2026-07-30, RTX 4090) -- Norman complete (102/107 conditions, 5 skipped for the documented zero-co-occurring-cells limitation), Adamson now also complete (21/21 conditions -- the 9 originally skipped by GPU-memory contention were re-run successfully once the contending process freed memory, see "Working conventions" below for a gotcha hit doing that re-run); Stage 4 (conventional metrics) implemented and run for real on both datasets for all five expression-predicting models (no-change, mean, ridge, GEARS, scGPT -- Geneformer excluded per its own documented exception); Stages 5–6 not yet started.** `docs/BUILD_PLAN.md` is the full project specification. All conda environments, the multi-stage `Dockerfile`, and the CI workflow exist. `src/stage3_models/train_scgpt.py` is a close port of `bowang-lab/scGPT`'s own `Tutorial_Perturbation.ipynb`; it has now actually been run end-to-end on an RTX 4090 for both Adamson and Norman (see its "Working conventions" entry below for real metrics) — treat it the same as `train_gears.py`, a known-working script, not an unverified port. `src/stage3_models/geneformer_wrapper.py` + `run_geneformer.py` were a riskier port than scGPT's (written against Geneformer's own source code directly, not a complete official tutorial) and that risk materialized: a small-subset validation run (2026-07-30, 7 real held-out Adamson `unseen_single` test conditions, `max_ncells=200`, RTX 4090) surfaced and fixed five real bugs (`emb_mode`, GPU OOM, missing output dirs, a library function with no `return` statement, and a multi-gene/"combo" stats-aggregation crash — full detail in "Working conventions" below) before it produced plausible, non-degenerate cosine-shift output. That multi-gene fix was then re-validated the same day against 6 real Norman test conditions (3 combo, spanning `combo_seen0/1/2`, plus 3 single-gene) — the combo perturbations succeeded outright, genuinely confirming the fix on real Norman data, but a sixth bug (a shared-output-directory issue in `run_geneformer.py` itself, unrelated to the multi-gene fix) surfaced and was fixed before the full 6-condition run passed cleanly. A separate, real (non-bug) limitation was also found: some gene pairs have zero control cells expressing both genes, which still crashes `run_geneformer.py`'s per-condition loop with no recovery — open, see "Working conventions" below. `src/stage4_conventional_metrics/` now has real code (`metrics.py`, `run_baselines_gears.py`, `run_scgpt.py`, `combine_results.py` -- see its own "Working conventions" entry below for real results); `src/stage5_calibration/` and `src/stage6_reporting/` are still empty package stubs (directory + `__init__.py` only) — that's intentional scaffolding matching `docs/BUILD_PLAN.md` §7, not partial implementation. Fill each in when that stage actually starts, and update this file's commands as you go.

## Commands

```
# Run the fast unit tests (no network) — what CI's unit-tests job runs
pytest tests/ -m "not integration" -v

# Run everything including real GEARS-download/fit/train integration tests
pytest tests/ -v

# Stage 1 — download + split a dataset (writes into ./data)
python -m src.stage1_data.cli --dataset adamson
python -m src.stage1_data.cli --dataset norman

# Stage 3a — REAL GEARS training run (GEARS's own default hyperparameters:
# hidden_size=64, epochs=20 -- see configs/training_hyperparameters.yaml).
# Distinct from tests/test_stage3a_gears.py's tiny smoke-test config.
# Measured ~15-17 hours on a laptop CPU (i7-10510U) for Adamson -- use a
# CUDA GPU instead (see environment-gears-gpu.yml / scripts/run_gears_gpu.sh,
# tested against an RTX 3090):
python -m src.stage3_models.train_gears --dataset adamson
python -m src.stage3_models.train_gears --dataset norman

# Stage 3b — scGPT fine-tuning (needs the perturb-bench-scgpt env, a GPU,
# and the pretrained checkpoint -- see Environment setup below). Verified
# end-to-end on an RTX 4090 for both datasets -- see Working conventions
# below for real metrics:
python scripts/download_scgpt_checkpoint.py
python -m src.stage3_models.train_scgpt --dataset adamson
python -m src.stage3_models.train_scgpt --dataset norman

# Stage 3c — Geneformer zero-shot in silico perturbation (needs the
# perturb-bench-geneformer env, a GPU, and a git-lfs clone of the pretrained
# checkpoint -- see Environment setup below). No training loop (zero-shot) --
# outputs a per-condition cosine-shift ranking CSV, not an expression
# prediction; see geneformer_wrapper.py's docstring and BUILD_PLAN.md Sec.6
# Stage 3c for why. --model-dir is the cloned repo root itself, NOT a
# subdirectory -- the default checkpoint (Geneformer-V2-316M) ships its
# config.json/model.safetensors directly at the repo root. Validated on
# small real subsets of BOTH datasets (RTX 4090): 7 Adamson conditions, then
# 6 Norman conditions (3 combo -- one each from combo_seen0/1/2 -- plus 3
# single-gene) once the Adamson-only run had shaken out enough bugs. Six
# real bugs total found and fixed across both -- see Working conventions
# below. Full-dataset (all conditions, not just a hand-picked subset) runs
# have now been done for real on both datasets (2026-07-30, RTX 4090) --
# Norman finished cleanly: 102/107 conditions, the other 5 skipped for the
# already-documented zero-co-occurring-cells combo limitation (see Working
# conventions below). Adamson only got 12/21 done -- the other 9 failed on
# CUDA OOM caused by an unrelated process from another user contending for
# the same GPU, not a code or data bug -- re-run the missing 9 (via
# --conditions, see Working conventions below for the exact list) once the
# GPU is free:
git lfs install && git clone https://huggingface.co/ctheodoris/Geneformer
python -m src.stage3_models.run_geneformer --dataset adamson --model-dir ./Geneformer
python -m src.stage3_models.run_geneformer --dataset norman --model-dir ./Geneformer

# Stage 4 -- conventional metrics (MSE@20DEG, Pearson(delta control)) for
# every model that predicts a per-gene expression profile (no-change, mean,
# ridge, GEARS, scGPT -- NOT Geneformer, which has no such prediction, see
# its Stage 3c note above). Two driver scripts because baselines/GEARS and
# scGPT need incompatible cell-gears versions (see Environment setup below)
# -- metrics.py itself has no gears/scgpt import, so both share the same
# MSE@20DEG/Pearson(delta control) code. Verified for real on both datasets
# -- see Working conventions below for the actual numbers and two
# methodological findings (a cross-env test-split mismatch, and why
# no-change's Pearson is NaN by construction):
conda activate perturb-bench-gears-gpu  # or perturb-bench/base once created
python -m src.stage4_conventional_metrics.run_baselines_gears --dataset adamson
python -m src.stage4_conventional_metrics.run_baselines_gears --dataset norman
conda activate perturb-bench-scgpt
python -m src.stage4_conventional_metrics.run_scgpt --dataset adamson
python -m src.stage4_conventional_metrics.run_scgpt --dataset norman
conda activate perturb-bench-gears-gpu  # combine_results.py just needs pandas -- any env with it works
python -m src.stage4_conventional_metrics.combine_results
```

Stage 2 has no CLI of its own by design — it's importable classes exercised via tests, now also driven end-to-end by Stage 4's `run_baselines_gears.py` (see Commands above):
- `src/stage2_baselines/` — `no_change.py`, `mean_baseline.py`, `ridge_baseline.py`: `fit(adata, train_conditions)` / `predict(condition)`, tested in `tests/test_stage2_baselines.py`.

`src/stage3_models/gears_wrapper.py` — `GearsModel`: `fit(pert_data)` / `predict(condition)` (also `save(path)`/`load(pert_data, path)`), tested in `tests/test_stage3a_gears.py` (fast unit tests + a smoke-scale integration test) and driven for real by `src/stage3_models/train_gears.py` (see Commands above). Note `fit()`'s different signature from Stage 2 — see "Working conventions" below for why.

`src/stage3_models/train_scgpt.py` is deliberately **not** a `fit()`/`predict()` class like `GearsModel` — scGPT's own documented fine-tuning procedure (`Tutorial_Perturbation.ipynb`) is a linear training script (custom train loop, AMP, early stopping, its own eval/plotting), not something that fits Stage 2's baseline contract without inventing an interface scGPT itself doesn't have. It's a script, run via the CLI above, same shape as `train_gears.py` — see "Working conventions" below.

`src/stage3_models/geneformer_wrapper.py` is **not** a `fit()`/`predict()` class either, one level further than `train_scgpt.py`: it's zero-shot (nothing to `fit()`) and its output is a per-gene cosine-shift ranking, not an `np.ndarray` expression profile, so there's no `predict(condition)` to write in the first place. It exposes plain functions (`add_geneformer_columns`, `symbols_to_ensembl`, `tokenize_control_cells`, `run_in_silico_perturbation`, `compute_perturbation_stats`) driven by the `run_geneformer.py` script (see Commands above) — see the Tech stack and Working conventions sections below for the real API constraints this was verified against.

All of the above assume the right conda env is active for the stage (see Environment setup below) — e.g. `gears`/`torch_geometric` aren't installed outside `perturb-bench`, `scgpt` isn't installed outside `perturb-bench-scgpt`, so these commands will import-error in the wrong env or a bare system Python.

## Environment setup

Separate conda environments — do not merge them, see below for why.

```
# Base — Stages 1, 2, 3a, 4, 5, 6 (CPU; also what CI runs)
conda env create -f environment.yml && conda activate perturb-bench
# or, pip-only (CI / Docker):
pip install -r requirements.txt

# Stage 3a on a GPU machine (optional -- only for running train_gears.py's
# real, non-smoke training run at practical speed; CI/the base env stay
# CPU-only). Tested against an RTX 3090; see scripts/run_gears_gpu.sh for a
# one-command wrapper.
conda env create -f environment-gears-gpu.yml && conda activate perturb-bench-gears-gpu

# Stage 3b — scGPT fine-tuning (GPU; tested against an RTX 3090, Ampere).
# flash-attn is enabled by default in this env file (Ampere+ supports it) --
# see scripts/run_scgpt_gpu.sh for a one-command wrapper that also fetches
# the pretrained checkpoint via scripts/download_scgpt_checkpoint.py.
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
- **`data-smoke-test`** (depends on `unit-tests`, `timeout-minutes: 20` as a safety net) — runs `pytest tests/ -m integration -v` against real data: GEARS's actual download path (Stage 1), fitting all three baselines against the real split (Stage 2), and — as of Stage 3a — a minimal (`epochs=1`, `hidden_size=8`; see `configs/training_hyperparameters.yaml`'s `stage3a_gears.smoke_test`) real GEARS training run. `./data` is cached across runs by dataset name (the `data_dir` fixture in `tests/conftest.py` intentionally does *not* use pytest's `tmp_path`, so this cache actually hits). Dataset defaults to Adamson (smaller, single-gene) via the `PERTDATA_TEST_DATASET` env var, read by every integration test — trigger Norman via `workflow_dispatch` with `dataset: norman` when needed. New stages should just add their own `@pytest.mark.integration` tests; this job picks them up automatically, no `ci.yml` edits required. GEARS is the one DL model BUILD_PLAN.md §5 calls out for automatic CI smoke-testing (scaled down) — scGPT/Geneformer fine-tuning stay manual/on-demand, never wired into this job.

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
├── perturbation_conditions.py    # created — shared 'GENE+ctrl'/'GENE1+GENE2' condition-string parsing (used by ridge_baseline.py and gears_wrapper.py)
├── stage1_data/              # created — load_data.py (PertData wrapper), cli.py
├── stage2_baselines/         # created — no_change.py, mean_baseline.py, ridge_baseline.py
├── stage3_models/            # gears_wrapper.py + train_gears.py + train_scgpt.py + geneformer_wrapper.py + run_geneformer.py created
├── stage4_conventional_metrics/  # metrics.py + run_baselines_gears.py + run_scgpt.py + combine_results.py created
├── stage5_calibration/      # stub only
└── stage6_reporting/        # stub only
notebooks/                  # not yet created — 01_data_exploration, 02_baseline_vs_models_norman,
                             # 03_baseline_vs_models_adamson, 04_calibration_analysis
configs/
└── training_hyperparameters.yaml   # created — stage1_data:, stage2_baselines:, stage3a_gears:, stage3b_scgpt:, stage3c_geneformer: sections; add one per stage as it's built
scripts/
├── bootstrap_remote_gpu.sh       # created — single entry point for a fresh remote GPU box: installs git/conda if missing, clones the repo, runs both run_*_gpu.sh scripts below
├── run_gears_gpu.sh              # created — one-command wrapper around train_gears.py for a GPU machine (conda-env-create-if-missing + run)
├── run_scgpt_gpu.sh              # created — same, for train_scgpt.py (also fetches the pretrained checkpoint)
└── download_scgpt_checkpoint.py  # created — gdown-based fetch of scGPT's 'whole-human' pretrained checkpoint (Google Drive-hosted, no plain URL)
tests/
├── conftest.py                    # created — data_dir fixture (fixed ./data path, not tmp_path, so CI caching works)
├── test_perturbation_conditions.py  # created — pure-logic unit tests, no heavy deps needed
├── test_stage1_data.py            # created — unit tests + one @pytest.mark.integration real-download test
├── test_stage2_baselines.py       # created — toy-data unit tests (exact arithmetic checked) + one integration test on the real split
└── test_stage3a_gears.py          # created — fast RuntimeError-before-fit tests + one integration test (real minimal training run)
environment.yml / requirements.txt   # created — base CPU env, Stages 1/2/3a/4/5/6
environment-gears-gpu.yml            # created — optional GPU env for train_gears.py's real run at practical speed (not used by CI)
environment-scgpt.yml                # created — GPU env, Stage 3b
environment-geneformer.yml           # created — GPU env, Stage 3c
checkpoints/, save/, data_scgpt/     # gitignored — scGPT pretrained checkpoint, fine-tuning run outputs, and its own PertData cache (kept
                                      # separate from ./data since perturb-bench-scgpt pins an older, unverified-compatible cell-gears)
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
- `cell-gears` (confirmed current PyPI name/version: `0.1.2`) for GEARS. Requires PyTorch Geometric installed first (`torch_geometric`; optional accelerated ops like `torch-scatter` need wheels matched to the exact torch build). The `GEARS` model class API (verified directly against `gears/gears.py`/`gears/inference.py`, not guessed): `GEARS(pert_data, device=..., weight_bias_track=...)` → `.model_initialize(hidden_size=...)` → `.train(epochs=..., lr=..., weight_decay=...)` → `.predict([[gene, ...], ...])`, which returns a dict keyed by `"_".join(genes)` (underscore-joined, gene names only — **not** the `"+"`-joined, `ctrl`-inclusive format `adata.obs['condition']` uses). `src/perturbation_conditions.py` + `src/stage3_models/gears_wrapper.py` handle that translation.
- `scgpt` (from `github.com/bowang-lab/scGPT`) for scGPT fine-tuning — **must live in its own environment**, not `perturb-bench` (see Environment setup above for the `cell-gears` version conflict). Also pins `scanpy>=1.9.1,<2.0.0`, `scvi-tools>=0.16.0,<1.0`, `orbax<0.1.8`; `flash-attn` (enabled in `environment-scgpt.yml`, Ampere+ target) is needed for the fast attention path (GPU/CUDA-build-specific) that `configs/training_hyperparameters.yaml`'s `stage3b_scgpt.use_fast_transformer: true` turns on. `src/stage3_models/train_scgpt.py`'s data loading / model / train / eval logic (verified directly against `bowang-lab/scGPT`'s own `tutorials/Tutorial_Perturbation.ipynb`, not guessed) uses the *pretrained* `whole-human` checkpoint (`scripts/download_scgpt_checkpoint.py`, Google-Drive-hosted) whose `args.json` overrides the config's `embsize`/`d_hid`/`nlayers`/`nhead`/`n_layers_cls` fields at load time — those config values only apply if you deliberately train from scratch instead.
  **Real, verified incompatibility found running this on an RTX 4090, now fixed**: PyPI's `scgpt==0.2.4` `model.py` does `from flash_attn.flash_attention import FlashMHA` — an API that only ever existed in flash-attn 1.x. `pyproject.toml`'s own declared `[fa2]` extra (`flash-attn>=2.8.0`) installs a 2.x package that has no `flash_attn.flash_attention` module at all, so that import always fails regardless of environment setup — a bug in the upstream `scgpt` PyPI release itself (its pyproject.toml was updated for flash-attn 2.x but its model code wasn't). `scgpt`'s own `TransformerGenerator.__init__` catches the failed import and silently falls back to vanilla PyTorch attention (just a `UserWarning`), so `train_scgpt.py` doesn't crash outright -- but this project's original checkpoint-loading code (a naive `startswith(prefix)` filter with no key-existence/shape check) then crashed on `model.load_state_dict()`, because the checkpoint's fused-QKV flash-attn parameter names (`self_attn.Wqkv.*`) don't exist on the fallback `nn.MultiheadAttention`-based model (which uses `self_attn.in_proj_*` instead).
  bowang-lab actually fixed the root `flash_attn.flash_attention` import upstream on GitHub `main` (PR #351, "cuda-12.8 and FlashAttention 2.x compatibility", commit `cebd6fae655b9c585a4807daa3ac31bb764f06b4`, merged 2026-04-27 -- unreleased to PyPI as of this writing) by adding `scgpt/model/flash_attn_compat.py`, and separately added `scgpt.utils.load_pretrained(model, pretrained_params, strict=False, prefix=[...])` (`scgpt/utils/util.py`), which renames checkpoint keys (`self_attn.Wqkv.` -> `self_attn.in_proj_` when flash-attn isn't in use, or to flash-attn's own wrapped naming when it is) before filtering to prefix+key+shape matches. **Both are now in use here**: `environment-scgpt.yml` installs `scgpt` from that pinned GitHub commit instead of PyPI, and `train_scgpt.py` calls `scgpt.utils.load_pretrained(...)` instead of the old hand-rolled filter -- this correctly loads the pretrained transformer attention weights even without a working flash-attn (`nn.MultiheadAttention.in_proj_weight` has the same fused-QKV shape as flash-attn's `Wqkv`), so fine-tuning genuinely starts from the pretrained checkpoint, not a partially-random model. Don't revert to PyPI's `scgpt` or the old loading code without re-verifying this.
- `Geneformer` (git-lfs clone from Hugging Face, `ctheodoris/Geneformer`, then `pip install .` — not on PyPI; may require gated/credentialed HF access, confirm before use) for zero-shot in silico perturbation. Requires Python >=3.10; deps (`transformers==4.46`, `peft`, `ray`, `optuna`, etc.) are permissive enough to coexist with the base env's pins, but keep it in its own environment anyway for the same GPU-isolation reason as scGPT. Pretrained checkpoint: **`Geneformer-V2-316M`** — confirmed by actually cloning the repo and reading its live README ("The current default model in the main directory of the repository is Geneformer-V2-316M"); its `config.json`/`model.safetensors` sit directly at the repo root (18 layers, hidden_size 1152, 4096-token context via `model_input_size`), not under a named subdirectory, so `--model-dir` should point at the cloned repo root itself. An earlier note here said `gf-12L-95M-i4096`, which does not exist anywhere in the current repo — that was a web-search-derived guess, never checked against the actual clone; don't trust web-search summaries over cloned-repo state for this kind of fact again. (`Geneformer-V2-104M`, 12 layers/768 hidden, also exists in the clone as a smaller alternative checkpoint if `V2-316M`'s memory footprint is ever a problem — see the `forward_batch_size` OOM note below.)
  **`InSilicoPerturber`/`InSilicoPerturberStats` API (verified directly against `geneformer/in_silico_perturber.py` and `geneformer/in_silico_perturber_stats.py` source, not guessed) fundamentally does not produce a per-gene expression prediction** — its public output is a cosine-similarity shift between a cell's embedding before/after a simulated perturbation, aggregated per gene into a ranking (`InSilicoPerturberStats.get_stats()`'s CSV). This is why `geneformer_wrapper.py` has no `predict(condition) -> np.ndarray` — there is no expression vector to return. Ahlmann-Eltze et al. 2025 got an MSE/Pearson-comparable number out of Geneformer by fitting a ridge-regression decoder on top of perturbed embeddings, but that needs undocumented internal token-manipulation/embedding-extraction functions with no official tutorial to verify a port against (checked: no such example exists in the repo's `examples/` directory or docs) — a materially higher risk than porting GEARS/scGPT, both done against complete official examples. **Decision (see `docs/BUILD_PLAN.md` §6 Stage 3c): use the public API as-is**, accepting that Stage 4/6 need a documented exception for Geneformer rather than one unified metrics table.
  **`genes_to_perturb` passed as a list perturbs every gene in it *together as one combined group* in every cell** (confirmed from `in_silico_perturber.py`'s `__init__`: a list sets `perturb_group=True` and force-resets `combos`/`anchor_gene`) — it does **not** test each gene individually. `run_geneformer.py` calls `run_in_silico_perturbation()` once per held-out test condition (a 1-element list for Adamson's single-gene knockdowns, a 2-element list for Norman's double-gene combos), not once for the whole test-gene set — passing the full test-gene list in one call would silently simulate one giant combined knockout instead of per-condition results.
  **No true raw per-cell counts are available for tokenization.** `TranscriptomeTokenizer` expects `adata.obs['n_counts']` to be each cell's real, varying total UMI count, but GEARS's `perturb_processed.h5ad` (this project's only Stage 1 data source) ships only post-`normalize_total`+`log1p` expression — confirmed directly against `data/adamson/perturb_processed.h5ad`/`data/norman/perturb_processed.h5ad` (`.raw is None`, empty `.layers`). Reconstructing counts via `expm1(X).sum(axis=1)` gives a near-constant ~5000–5200 per cell (std ~3% of the mean, checked on Adamson) rather than real varying sequencing depth — the true depth signal was normalized away upstream and isn't recoverable from this file. `geneformer_wrapper.py` uses that reconstruction anyway as a documented approximation (total-count normalization is a per-cell scalar multiply, so it preserves each cell's *within-cell* gene rank order, the dominant signal the tokenizer's rank-value encoding uses — see the module's docstring for the full reasoning and caveat). If Stage 3c's first real results look implausible, check this before assuming the model or code is at fault.
- `scikit-learn` `Ridge` for the baseline.
- GitHub Actions CI smoke-tests only the baseline + GEARS pipeline on a small subset (installs from `requirements.txt`) — foundation-model fine-tuning is too heavy for CI and is run manually/on-demand.

## Working conventions specific to this project

- **Sanity-check the no-change baseline first**, before building anything else. If it doesn't land in a plausible range vs. published numbers, debug the pipeline before proceeding — it's the field's own headline empirical anchor.
- **Cross-check GEARS output against its own published Norman benchmark numbers** (e.g. ~0.176 MSE@20DEG in one commonly cited comparison) as a pipeline-correctness check before trusting scGPT/Geneformer runs.
- **Separate "doesn't match the paper" (debugging problem) from "shows a genuinely different pattern" (real finding)** — document either outcome; never silently adjust methodology until numbers look "right."
- **Run and report both Norman (combinatorial) and Adamson (single-gene)** — a conclusion that differs between the "easy" and "hard" perturbation regime is a legitimate finding to publish, not a discrepancy to paper over.
- **Every hyperparameter and random seed goes in `configs/`** — reproducibility by an independent party is a hard project requirement, not a nice-to-have, given the benchmarking-fragility literature this project is directly engaging with.
- Metric definitions must match the source papers exactly (MSE@20DEG, Pearson of delta-from-control via t-test DEG selection) — don't substitute a "close enough" metric variant.
- **Stage 2's baseline interface is the contract later stages should match**: `fit(adata: AnnData, train_conditions: list[str])` / `predict(condition: str) -> np.ndarray` (a per-gene mean expression profile, `n_genes` long). All three baselines share it — even `NoChangeBaseline`, which ignores `train_conditions` — specifically so Stage 4's evaluation loop can treat every baseline uniformly. `RidgeBaseline` uses a multi-hot gene-identity encoding as its "perturbation embedding," a deliberate, documented simplification of the ESM-2-style embeddings BUILD_PLAN.md §5 mentions — not a hidden shortcut.
- **`GearsModel` (Stage 3a) deliberately does *not* match that exact signature**: `fit(pert_data)` takes the whole `PertData` object, not `(adata, train_conditions)`, because GEARS's model genuinely needs the graph-structured dataloaders `PertData.get_dataloader()` builds (GO similarity / gene co-expression graphs) — a bare `AnnData` can't provide that. `predict(condition) -> np.ndarray` still matches Stage 2's output shape, and `predict_many(conditions) -> dict[str, np.ndarray]` exists for the batched case (GEARS's own `predict()` is natively batched; calling it once per condition would rebuild a DataLoader every time). This is a real difference in what the model needs, not an inconsistency to paper over — Stage 4 will need to branch on it, not force a fake-uniform interface.
- **GEARS's own hyperparameter defaults are used as-is** (`hidden_size=64, epochs=20, lr=1e-3, weight_decay=5e-4`) per BUILD_PLAN.md §6 Stage 3a ("default hyperparameters," not tuned). The CI/integration smoke test uses a separate, much smaller `stage3a_gears.smoke_test` config (`epochs=1, hidden_size=8`) purely to confirm the training loop runs — never conflate the two when reading results.
- **`train_gears.py`'s real (non-smoke) run needs the same memory care as the smoke test did before it was fixed**: `PertData.load()` builds one graph per single cell for the *entire* dataset regardless of hyperparameters, and this is genuinely slow on CPU too — measured ~15-17 hours projected for Adamson's 20 epochs on a laptop CPU (i7-10510U). `environment-gears-gpu.yml` / `scripts/run_gears_gpu.sh` exist to run this on a separate CUDA GPU machine instead (target hardware: an RTX 3090). **Now actually verified**: run for real on an RTX 4090 (24GB). Adamson's 20 epochs completed in 1309.5s (~22min) — `Test Top 20 DE MSE: 0.2364`, `test_unseen_single_pearson: 0.988`, `test_unseen_single_pearson_de: 0.960` (the `combo_seen0/1/2` metrics are `nan` by design: Adamson has no combinatorial perturbations, only `unseen_single`). Checkpoint saved to `./data/gears_adamson_real_ckpt`. Norman's 20 epochs completed in 1541.6s (~26min) — `Test Top 20 DE MSE: 0.1683`, matching this same file's own cited published anchor (~0.176) closely, a strong pipeline-correctness confirmation per this section's own cross-check convention. Checkpoint saved to `./data/gears_norman_real_ckpt`.
- **`train_scgpt.py` is now verified end-to-end**, run for real on an RTX 4090 (24GB) after fixing the flash-attn/checkpoint-loading issue documented above. Both runs used `epochs=15` (early stopping triggered before the configured max) with the default `stage3b_scgpt` hyperparameters. Adamson: best val pearson 0.9906 (epoch 14); test set `test_unseen_single_pearson_top20_de_non_dropout: 0.976`, `test_unseen_single_pearson_delta_de: 0.794` (Adamson has only `unseen_single` perturbations, no combinatorial ones, so `combo_seen0/1/2` metrics are `nan` by design, same as GEARS). Run dir: `save/dev_perturb_adamson-Jul30-03-51/`. Norman: best val pearson 0.9909 (epoch 14); test set `test_unseen_single_pearson_delta_de: 0.525`, and `combo_seen0/1/2_pearson_delta_de: 0.268 / 0.760 / 0.882` — increasing with more seen genes in the combo, as expected. Run dir: `save/dev_perturb_norman-Jul30-07-55/`. Both `best_model.pt` checkpoints and `run.log`/`test_metrics.json` are saved under `save/` (gitignored). Treat it as equivalent in reliability to `train_gears.py` now — no longer an unverified port.
- **`geneformer_wrapper.py`/`run_geneformer.py` were UNTESTED end-to-end and a step riskier than `train_scgpt.py`'s original port — that risk materialized.** `train_gears.py`/`train_scgpt.py` were both ported against a complete official tutorial covering the exact usage pattern needed; no equivalent official example exists for "zero-shot in silico perturbation restricted to a specific test-gene list, aggregated per condition" — this was written directly against `geneformer/tokenizer.py`, `geneformer/in_silico_perturber.py`, and `geneformer/in_silico_perturber_stats.py` source (real signatures, not guessed). **Validated 2026-07-30 on an RTX 4090** against 7 real held-out Adamson `unseen_single` test conditions (`CREB1`, `BHLHE40`, `DDIT3`, `HARS`, `DAD1`, `DDRGK1`, `SAMM50` — the first 7 of the real `simulation`-split test set per `configs/training_hyperparameters.yaml`'s `stage1_data`, not a hand-picked or synthetic subset), `max_ncells=200` for smoke-test speed. Four real bugs surfaced and were fixed, in the order hit:
  1. **`emb_mode: cell` crashes on V2 checkpoints**: V2 tokenizes with a leading `<cls>` token, and `InSilicoPerturber.perturb_data` hard-requires `emb_mode` to contain `"cls"` whenever the first token is `<cls>` (`in_silico_perturber.py` ~line 479), raising `"Emb mode 'cls' or 'cls_and_gene' required when first token is <cls>."` otherwise. Fixed: `stage3c_geneformer.emb_mode` in the config and `run_in_silico_perturbation`'s default both changed `cell` → `cls` (the library's own upstream default, which the original config had overridden without reason).
  2. **`forward_batch_size: 100` OOMs a 24GB RTX 4090** (`torch.OutOfMemoryError`, tried to allocate 11GiB). Root cause: `get_embs` requests `output_hidden_states=True` (needed so it can select `layer_to_quant`), which makes the model retain all 19 layers' `batch × seq × hidden` activations simultaneously — at `batch=100`, Adamson control-cell sequences (tokenized length: median 1046, max 1587, well under the 4096 cap) already cost ~13–15GB just for that, before the `BertForMaskedLM` head's `batch × seq × vocab_size(20275)` output tensor adds more on top. Fixed: `forward_batch_size` dropped to `16` (config + wrapper default), sized to stay under budget even at the full 4096-token cap (see the config's own comment for the arithmetic) — this is the real, verified-safe number for `Geneformer-V2-316M` on a 24GB card, not a guess.
  3. **`run_in_silico_perturbation`/`compute_perturbation_stats` never created their `output_directory`** (unlike `tokenize_control_cells`, which does), so `InSilicoPerturber.perturb_data` crashed with `FileNotFoundError` partway through the first condition, mid-write. Fixed: both functions now `Path(output_directory).mkdir(parents=True, exist_ok=True)` before calling into the library, matching `tokenize_control_cells`'s existing pattern.
  4. **`InSilicoPerturberStats.get_stats()` has no `return` statement** in the installed version — it only calls `cos_sims_df.to_csv(...)` and falls off the end, so `compute_perturbation_stats` was silently getting `None` back (an incorrect assumption in its own original docstring, never actually verified against the installed library). `run_geneformer.py`'s `df["condition"] = condition` then crashed with `TypeError: 'NoneType' object does not support item assignment`. Fixed: `compute_perturbation_stats` now re-reads the CSV `get_stats()` writes to disk via `pd.read_csv(...)` instead of trusting a return value.

  After all four fixes, the run completed end-to-end and produced `data/geneformer_adamson_validation/adamson_geneformer_cosine_shift.csv`: 1400 rows (200 cells × 7 conditions), zero `NaN`/`inf`, no degenerate all-0 or all-1 values, real per-condition variation (`Cosine_sim` mean ranges ~0.99991–0.99995 across the 7 conditions, std ~2e-5 within each) — small shifts, but that's the expected order of magnitude for a single-gene deletion out of ~1000+ expressed genes in a high-dimensional cell embedding, not a sign of a broken/degenerate signal. The `expm1(X).sum(axis=1)` `n_counts` reconstruction was separately re-verified directly against real Adamson control cells: finite, all positive, range ~4526–8305 (mean ~5282), consistent with this file's earlier documented approximation.

  5. **`InSilicoPerturberStats`'s `aggregate_data`-mode helper (`isp_aggregate_grouped_perturb`, `in_silico_perturber_stats.py` ~line 230) crashes on multi-gene ("combo") groups** — a real bug in the installed `geneformer` library's own stats code, not this project's wrapper logic. `genes_to_perturb` as a list is confirmed (both by inspecting `InSilicoPerturber`'s own `perturb_group`/`combos`/`anchor_gene` attributes after construction, and by an actual run producing one pickle keyed `((tok1, tok2), 'cell_emb')`) to perturb multiple genes as one combined group, exactly as documented — the perturbation mechanism itself is correct for Norman's real 2-gene `combo_seen0/1/2` conditions. But `isp_aggregate_grouped_perturb` does `df["Gene"] = symbol` where, for a multi-gene group, `symbol` is itself a tuple (e.g. `('CREB1', 'DAD1')`) — pandas then tries to broadcast that 2-element tuple across the DataFrame's per-cell rows and raises `ValueError: Length of values (2) does not match length of index (N)`. Adamson has no combo conditions, so the main 7-condition validation never hit this; it was caught by an isolated synthetic 2-gene repro (two Adamson genes forced into one combined group). **Fixed**: `geneformer_wrapper.py` defines `_isp_aggregate_grouped_perturb_fixed`, an exact copy of the installed `isp_aggregate_grouped_perturb` except the final line is `df["Gene"] = [symbol] * len(df)` instead of `df["Gene"] = symbol` (a list assigns as one repeated scalar per row instead of being unpacked elementwise) -- since that's a no-op for the single-gene case (`[symbol] * len(df)` and pandas's own scalar broadcast produce an identical column), `compute_perturbation_stats` monkeypatches `geneformer.in_silico_perturber_stats.isp_aggregate_grouped_perturb` with it unconditionally, not just for multi-gene groups. Verified on both the original synthetic 2-gene repro (now produces 30 correct per-cell rows instead of crashing, `Gene` column holding the combined `('CREB1', 'DAD1')` group) and a single-gene regression check (unchanged output vs. pre-patch behavior). Monkeypatching the installed library from this project's own code (rather than hand-editing the installed package, or vendoring the whole file) keeps the fix reproducible from a fresh `pip install .` — verified against `ctheodoris/Geneformer` commit `04c2b2e84da7c0f385c3f9ad8f3ec24bab6650e5` (2026-05-26), the HEAD this project's local clone happened to be on; `environment-geneformer.yml` does **not** pin a commit (unlike `environment-scgpt.yml`, which does), so a fresh clone could silently land on a different commit — see the comment at the monkeypatch call site in `geneformer_wrapper.py` for what to re-check if that happens.

  **This fix was then re-validated for real** (2026-07-30, same RTX 4090): 6 real Norman `simulation`-split test conditions — `CBL+PTPN9` (`combo_seen0`), `DUSP9+MAPK1` (`combo_seen1`), `AHR+KLF1` (`combo_seen2`), `DUSP9+ctrl`/`BCORL1+ctrl`/`MEIS1+ctrl` (single-gene) — `max_ncells=200`. **All 3 combo conditions completed successfully**, including stats aggregation, genuinely confirming the fix works on real Norman combo data (not just the earlier synthetic repro): `data/geneformer_norman_validation/norman_geneformer_cosine_shift.csv` shows `CBL+PTPN9`'s `Gene` column correctly holding `('CBL', 'PTPN9')` across 200 rows, `DUSP9+MAPK1` correctly holding `('DUSP9', 'MAPK1')`, `AHR+KLF1` correctly holding `('AHR', 'KLF1')` across 13 rows (see bug 6b below for why only 13). All `Cosine_sim` values finite, no degenerate 0s/1s, real per-condition variation (means ranging ~0.99981–0.99994 across the 6 conditions). Two further findings came out of this Norman run, one a real bug (now fixed) and one a real, still-open limitation:

  6a. **`run_geneformer.py` shared one `isp_raw`/`isp_stats` directory across every condition in its per-condition loop** — a real bug in this project's own script, unrelated to the multi-gene fix itself. `InSilicoPerturberStats.get_stats` → `read_dictionaries` scans *every* file matching `*_raw.pickle` in `input_data_directory` by suffix alone, not filtered by `output_prefix` (confirmed directly against the installed `read_dictionaries` source) — so with one shared directory, each condition's stats call was silently re-reading every prior condition's pickle too. This was harmless (if wasteful) for the all-single-gene Adamson validation, since every accumulated dict key was a plain, mutually-sortable `int`, and `isp_aggregate_grouped_perturb`'s `Ensembl_ID` filtering still isolated the correct row regardless of the extra noise. It stopped being harmless once combo (tuple-keyed) and single-gene (int-keyed) pickles coexisted in the same directory: `get_gene_list`'s `gene_list.sort()` hard-crashed with `TypeError: '<' not supported between instances of 'int' and 'tuple'` processing the first single-gene condition after 3 combo conditions had already written pickles there. **Fixed**: `run_geneformer.py`'s per-condition loop now uses `isp_dir = f"{output_dir}/isp_raw/{prefix}"` / `stats_dir = f"{output_dir}/isp_stats/{prefix}"` (per-condition subdirectories) instead of one shared pair for the whole run — this also fixes the wasted-rescan inefficiency, not just the crash. (The earlier 7-condition Adamson validation's *results* are unaffected by this bug — its accumulated extra rows were always filtered out correctly by `Ensembl_ID`, just computed wastefully — but re-run any real Adamson production run with the fixed code anyway, since it's meaningfully faster per condition.)

  6b. **Some gene pairs have zero control cells expressing both genes simultaneously — a real, unfixed, non-bug limitation of "delete"-type perturbation, not a code defect.** The original combo_seen1 pick, `MAP2K6+SPI1`, crashed with `RuntimeError: No active exception to reraise` (a bare `raise` with no active exception in `geneformer/perturber_utils.py`'s `logging_filtered_data_len`, triggered when `filter_data_by_tokens_and_log`'s cell-filtering — which requires a cell to already tokenize *all* of a group's genes before it can simulate deleting them — returns zero cells). Directly confirmed by checking real token co-occurrence: of Norman's 7353 tokenized control cells, exactly 1 contains `MAP2K6` at all, and 0 contain both `MAP2K6` and `SPI1` together. This is expected: `MAP2K6` is rarely tokenized (i.e., rarely ranks within the model's `model_input_size`-token budget) in control cells, and "delete" perturbation is only defined for cells where the gene is already present to delete. `DUSP9+MAPK1` (589 co-occurring cells) was substituted as a working `combo_seen1` example instead — but the underlying issue is real and **is not specific to this one gene pair**: a real production run over all of Norman's 71 combo conditions could plausibly hit one or more pairs with this same zero-co-occurrence problem. **Now fixed** (was initially left as an open recommendation, then actually implemented): `run_geneformer.py`'s per-condition loop wraps each condition's `run_in_silico_perturbation`/`compute_perturbation_stats` pair in `try`/`except Exception`, logging the failure and appending it to a `{dataset}_geneformer_skipped_conditions.csv` rather than aborting the whole run; `Ctrl-C`/`SystemExit` still propagate normally since they aren't `Exception` subclasses. Verified against the real `MAP2K6+SPI1` failure case directly: it's caught, logged, and skipped while the rest of the run continues and still produces a combined CSV from the conditions that succeeded.

- **Full-dataset Geneformer production runs (2026-07-30, RTX 4090): Norman complete, Adamson partial.** With all six bugs above fixed, `run_geneformer.py` was run for real over the *entire* real test-condition set for both datasets (not a hand-picked subset) — `max_ncells=200` as before, no `--conditions`/`--max-conditions` restriction.
  - **Norman: finished cleanly, 102/107 conditions succeeded**, the other 5 (`MAP2K6+SPI1`, `TBX3+TBX2`, `PTPN13+ctrl`, `FOXF1+FOXL2`, `TGFBR2+PRTG`) all skipped with the already-documented `RuntimeError: No active exception to reraise` zero-co-occurring-cells combo limitation (see 6b above) — no new bugs, confirms that limitation generalizes to other gene pairs as predicted rather than being unique to `MAP2K6+SPI1`. Output: `data/geneformer_norman/norman_geneformer_cosine_shift.csv` (102 conditions), `norman_geneformer_skipped_conditions.csv` (5 conditions).
  - **Adamson: initially only 12/21 conditions succeeded** — the other 9 (`EIF2B2+ctrl`, `ARHGAP22+ctrl`, `SRP72+ctrl`, `TMED10+ctrl`, `COPB1+ctrl`, `FECH+ctrl`, `DHDDS+ctrl`, `CARS+ctrl`, `TMED2+ctrl`) all failed with `CUDA out of memory` errors. **Not a code or data bug**: the errors themselves cite a second process (PID 3844549, confirmed via a live `nvidia-smi` check to be an unrelated user's PyTorch job, `/home/akif/.conda/envs/whisper/bin/python`, a speech-separation benchmark) holding ~16–22GB of the shared RTX 4090's 23GB at the time — this project's own `forward_batch_size=16` sizing (see bug 2 above) assumed exclusive use of the card, which didn't hold here. **Since re-run successfully the same day once the contending process's usage dropped** (`nvidia-smi` showed it down to ~5.5GB before the retry) via `python -m src.stage3_models.run_geneformer --dataset adamson --model-dir ./Geneformer --conditions EIF2B2+ctrl,ARHGAP22+ctrl,SRP72+ctrl,TMED10+ctrl,COPB1+ctrl,FECH+ctrl,DHDDS+ctrl,CARS+ctrl,TMED2+ctrl` — all 9 completed cleanly this time, no OOM, no new bugs. Adamson Geneformer output is now genuinely complete: 21/21 conditions, `data/geneformer_adamson/adamson_geneformer_cosine_shift.csv` (284852 rows across all 21 conditions), no skipped-conditions file (nothing failed on the re-run, so `run_geneformer.py` never wrote one — the stale one from the earlier partial run was deleted rather than left to be misread as current).
  - **A real gotcha hit doing this re-run, now documented so it isn't repeated**: `run_geneformer.py`'s final combine step (`combined = pd.concat(all_stats, ...); combined.to_csv(combined_path, ...)`, ~line 201-203) only concatenates the *current invocation's* conditions and unconditionally overwrites `{dataset}_geneformer_cosine_shift.csv` — it does **not** merge with a prior run's output. Running `--conditions <the 9 missing ones>` therefore clobbered the original 12-condition CSV down to just the 9 new rows. **Recovered without data loss** only because bug 6a's per-condition `isp_stats/{prefix}/{prefix}.csv` subdirectories are untouched by later runs on *different* conditions — the original 12 conditions' stats files were still sitting on disk and were reconstructed into the correct 21-condition combined CSV by re-reading each `isp_stats/<condition_prefix>/<condition_prefix>.csv` directly and re-concatenating (prefix → condition via `prefix.replace("_ctrl", "+ctrl")`, matching `_safe_prefix`'s own `condition.replace("+", "_")` exactly in reverse). **Now fixed**: `run_geneformer.py`'s combine step reads any pre-existing `{dataset}_geneformer_cosine_shift.csv`/`{dataset}_geneformer_skipped_conditions.csv` first, drops only the rows for conditions this run actually produced a fresh *successful* result for (a condition that already succeeded in a prior run and happens to fail on a re-run keeps its old good data rather than losing it), and merges the rest back in -- verified both against a simulated repro of this exact scenario and by re-deriving the real 21-condition Adamson file from the surviving per-condition `isp_stats/` subdirectories to confirm the reconstructed output matched. A partial `--conditions` re-run no longer clobbers prior results.

- **Stage 4 (conventional metrics) is implemented and has been run for real on both datasets** (2026-07-30): `src/stage4_conventional_metrics/metrics.py` implements MSE@20DEG (top-20 DE genes per condition via `scanpy.tl.rank_genes_groups`, t-test, `rankby_abs=True` — matching GEARS's own `rank_genes_groups_by_cov` convention, see the module's own docstring for why an exact match to GEARS's self-reported number isn't expected) and Pearson(delta control) (correlation of predicted vs. real expression change from control, across all genes), behind one `evaluate_model(predict_fn, adata, test_conditions, model_name)` function every model runs through identically. Two driver scripts exist because baselines/GEARS and scGPT need incompatible `cell-gears` versions (see Environment setup): `run_baselines_gears.py` (env: `perturb-bench-gears-gpu`) fits the three Stage 2 baselines fresh and loads GEARS from its existing real checkpoints (`data/gears_{dataset}_real_ckpt`, no retraining); `run_scgpt.py` (env: `perturb-bench-scgpt`) loads the fine-tuned scGPT checkpoint (`save/dev_perturb_{dataset}-*/best_model.pt`, auto-picks the most recent one that actually has a checkpoint, skipping earlier aborted attempts) and reuses `train_scgpt.py`'s own `predict()` function unchanged, wrapped to match the uniform contract. `combine_results.py` merges both drivers' per-dataset CSVs into one summary.

  **Real results** (mean per model per dataset, `n` = number of real held-out test conditions actually evaluated):

  | dataset | model | mean MSE@20DEG | mean Pearson(Δcontrol) | n |
  |---|---|---|---|---|
  | adamson | no_change | 0.3800 | NaN (see below) | 21 |
  | adamson | mean | 0.2478 | 0.6357 | 21 |
  | adamson | ridge | 0.2482 | 0.6357 | 21 |
  | adamson | gears | 0.2465 | 0.5557 | 21 |
  | adamson | scgpt | **0.1240** | **0.6405** | 22 |
  | norman | no_change | 0.4368 | NaN (see below) | 107 |
  | norman | mean | 0.3316 | 0.5786 | 107 |
  | norman | ridge | **0.2308** | **0.6732** | 107 |
  | norman | gears | 0.1689 | 0.5831 | 107 |
  | norman | scgpt | 0.2235 | 0.5505 | 116 |

  A genuinely interesting, dataset-dependent finding (exactly the kind of thing this project exists to surface honestly, per BUILD_PLAN.md §0–§2's framing): **scGPT clearly wins on Adamson** (single-gene perturbations) on both metrics, but **ridge — the simplest baseline — wins on Norman** (combinatorial perturbations) on both metrics, beating both DL models (GEARS and scGPT). Don't average these two datasets together into one number when reporting — the ranking genuinely flips between them.

  **Sanity check per BUILD_PLAN.md §9** (compare against each model's own independently-reported real numbers, not expecting exact agreement — see metrics.py's docstring for why): GEARS's self-reported "Test Top 20 DE MSE" was 0.2364 (Adamson) / 0.1683 (Norman) vs. this Stage 4 pipeline's independently-computed 0.2465 / 0.1689 — close on both, particularly Norman (within 0.4%). scGPT's self-reported `test_metrics.json` `pearson_delta` was 0.6405064566059552 (Adamson) / 0.549310042047368 (Norman) vs. this pipeline's 0.6405477300286293 / 0.5504977556330890 — both matching almost to the decimal place, since these use the exact same "delta from control, correlation across all genes" definition (`scgpt.utils.compute_perturbation_metrics`'s `pearson_delta` and this project's `pearson_delta_control` are the same metric, computed two different ways). Both cross-checks strongly support Stage 4's own MSE@20DEG/Pearson(delta control) implementation being correct.

  **Two real findings surfaced building this, both worth knowing before trusting this table blindly:**
  1. **`no_change`'s Pearson(delta control) is mathematically `NaN`, not a bug.** `NoChangeBaseline.predict()` always returns the control mean itself, so its predicted delta-from-control is the zero vector for every condition — Pearson correlation of a constant (zero-variance) vector against anything is undefined. This is the correct, expected mathematical behavior of a "predict zero effect" baseline, not something to special-case away.
  2. **scGPT is evaluated on a genuinely different real test split than baselines/GEARS.** `perturb-bench-scgpt` pins `cell-gears==0.0.2`; `perturb-bench-gears-gpu` uses `cell-gears==0.1.2` (see Environment setup). Both call `PertData.prepare_split(split="simulation", seed=1)` identically, but the two library versions produce **different real train/test splits** from the same seed — confirmed directly by comparing the actual condition lists, not just counts: Adamson is 22 test conditions in the scGPT env vs. 21 in the gears-gpu env (only ~6 conditions actually overlap between the two lists); Norman is 116 vs. 107. This table reports each model against its own real, correct split rather than restricting to the intersection — a deliberate choice (the alternative, forcing all five models onto one canonical subset, is a legitimate option but wasn't taken here) — so scGPT's row is not computed over byte-identical conditions to the other four rows. Keep this in mind before reading small cross-model differences as more significant than they are; it doesn't affect any model's own internal correctness, and the Adamson/Norman ranking finding above is large enough to survive this caveat either way.

  All four CSVs (`data/stage4_results/{adamson,norman}_{baselines_gears,scgpt}.csv`), the merged per-condition table (`all_per_condition.csv`), and the summary above (`combined_summary.csv`) are saved under `data/stage4_results/` (gitignored, like the rest of `data/`).

## Licensing checklist to keep in mind

Before publishing any derived outputs, GEARS/Norman/Adamson redistribution terms, scGPT and Geneformer weight licenses (Geneformer may be gated on Hugging Face), and scPerturb terms of use (if used) all need confirming — see `docs/BUILD_PLAN.md` §11. Ahlmann-Eltze et al. 2025 and the rebuttal preprint must be cited prominently since the entire project framing depends on transparently building on both.
