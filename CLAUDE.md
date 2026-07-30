# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

**Stages 1–3b (data acquisition, baselines, GEARS, scGPT) implemented and verified with real runs; Stages 3c–6 not yet started.** `docs/BUILD_PLAN.md` is the full project specification. All conda environments, the multi-stage `Dockerfile`, and the CI workflow exist. `src/stage3_models/train_scgpt.py` is a close port of `bowang-lab/scGPT`'s own `Tutorial_Perturbation.ipynb`; it has now actually been run end-to-end on an RTX 4090 for both Adamson and Norman (see its "Working conventions" entry below for real metrics) — treat it the same as `train_gears.py`, a known-working script, not an unverified port. `src/stage3_models/geneformer_wrapper.py` still doesn't exist; `src/stage4_conventional_metrics/` through `src/stage6_reporting/` are empty package stubs (directory + `__init__.py` only) — that's intentional scaffolding matching `docs/BUILD_PLAN.md` §7, not partial implementation. Fill each in when that stage actually starts, and update this file's commands as you go.

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
```

Stage 2 has no CLI of its own by design — it's importable classes exercised via tests, not yet wired into an end-to-end runner:
- `src/stage2_baselines/` — `no_change.py`, `mean_baseline.py`, `ridge_baseline.py`: `fit(adata, train_conditions)` / `predict(condition)`, tested in `tests/test_stage2_baselines.py`.

It'll get driven end-to-end once Stage 4 (metrics) or Stage 6 (reporting) exists to actually consume its predictions.

`src/stage3_models/gears_wrapper.py` — `GearsModel`: `fit(pert_data)` / `predict(condition)` (also `save(path)`/`load(pert_data, path)`), tested in `tests/test_stage3a_gears.py` (fast unit tests + a smoke-scale integration test) and driven for real by `src/stage3_models/train_gears.py` (see Commands above). Note `fit()`'s different signature from Stage 2 — see "Working conventions" below for why.

`src/stage3_models/train_scgpt.py` is deliberately **not** a `fit()`/`predict()` class like `GearsModel` — scGPT's own documented fine-tuning procedure (`Tutorial_Perturbation.ipynb`) is a linear training script (custom train loop, AMP, early stopping, its own eval/plotting), not something that fits Stage 2's baseline contract without inventing an interface scGPT itself doesn't have. It's a script, run via the CLI above, same shape as `train_gears.py` — see "Working conventions" below.

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
├── stage3_models/            # gears_wrapper.py + train_gears.py + train_scgpt.py created; geneformer_wrapper.py still to go
├── stage4_conventional_metrics/  # stub only
├── stage5_calibration/      # stub only
└── stage6_reporting/        # stub only
notebooks/                  # not yet created — 01_data_exploration, 02_baseline_vs_models_norman,
                             # 03_baseline_vs_models_adamson, 04_calibration_analysis
configs/
└── training_hyperparameters.yaml   # created — stage1_data:, stage2_baselines:, stage3a_gears:, stage3b_scgpt: sections; add one per stage as it's built
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
- **Stage 2's baseline interface is the contract later stages should match**: `fit(adata: AnnData, train_conditions: list[str])` / `predict(condition: str) -> np.ndarray` (a per-gene mean expression profile, `n_genes` long). All three baselines share it — even `NoChangeBaseline`, which ignores `train_conditions` — specifically so Stage 4's evaluation loop can treat every baseline uniformly. `RidgeBaseline` uses a multi-hot gene-identity encoding as its "perturbation embedding," a deliberate, documented simplification of the ESM-2-style embeddings BUILD_PLAN.md §5 mentions — not a hidden shortcut.
- **`GearsModel` (Stage 3a) deliberately does *not* match that exact signature**: `fit(pert_data)` takes the whole `PertData` object, not `(adata, train_conditions)`, because GEARS's model genuinely needs the graph-structured dataloaders `PertData.get_dataloader()` builds (GO similarity / gene co-expression graphs) — a bare `AnnData` can't provide that. `predict(condition) -> np.ndarray` still matches Stage 2's output shape, and `predict_many(conditions) -> dict[str, np.ndarray]` exists for the batched case (GEARS's own `predict()` is natively batched; calling it once per condition would rebuild a DataLoader every time). This is a real difference in what the model needs, not an inconsistency to paper over — Stage 4 will need to branch on it, not force a fake-uniform interface.
- **GEARS's own hyperparameter defaults are used as-is** (`hidden_size=64, epochs=20, lr=1e-3, weight_decay=5e-4`) per BUILD_PLAN.md §6 Stage 3a ("default hyperparameters," not tuned). The CI/integration smoke test uses a separate, much smaller `stage3a_gears.smoke_test` config (`epochs=1, hidden_size=8`) purely to confirm the training loop runs — never conflate the two when reading results.
- **`train_gears.py`'s real (non-smoke) run needs the same memory care as the smoke test did before it was fixed**: `PertData.load()` builds one graph per single cell for the *entire* dataset regardless of hyperparameters, and this is genuinely slow on CPU too — measured ~15-17 hours projected for Adamson's 20 epochs on a laptop CPU (i7-10510U). `environment-gears-gpu.yml` / `scripts/run_gears_gpu.sh` exist to run this on a separate CUDA GPU machine instead (target hardware: an RTX 3090). **Now actually verified**: run for real on an RTX 4090 (24GB). Adamson's 20 epochs completed in 1309.5s (~22min) — `Test Top 20 DE MSE: 0.2364`, `test_unseen_single_pearson: 0.988`, `test_unseen_single_pearson_de: 0.960` (the `combo_seen0/1/2` metrics are `nan` by design: Adamson has no combinatorial perturbations, only `unseen_single`). Checkpoint saved to `./data/gears_adamson_real_ckpt`. Norman's 20 epochs completed in 1541.6s (~26min) — `Test Top 20 DE MSE: 0.1683`, matching this same file's own cited published anchor (~0.176) closely, a strong pipeline-correctness confirmation per this section's own cross-check convention. Checkpoint saved to `./data/gears_norman_real_ckpt`.
- **`train_scgpt.py` is now verified end-to-end**, run for real on an RTX 4090 (24GB) after fixing the flash-attn/checkpoint-loading issue documented above. Both runs used `epochs=15` (early stopping triggered before the configured max) with the default `stage3b_scgpt` hyperparameters. Adamson: best val pearson 0.9906 (epoch 14); test set `test_unseen_single_pearson_top20_de_non_dropout: 0.976`, `test_unseen_single_pearson_delta_de: 0.794` (Adamson has only `unseen_single` perturbations, no combinatorial ones, so `combo_seen0/1/2` metrics are `nan` by design, same as GEARS). Run dir: `save/dev_perturb_adamson-Jul30-03-51/`. Norman: best val pearson 0.9909 (epoch 14); test set `test_unseen_single_pearson_delta_de: 0.525`, and `combo_seen0/1/2_pearson_delta_de: 0.268 / 0.760 / 0.882` — increasing with more seen genes in the combo, as expected. Run dir: `save/dev_perturb_norman-Jul30-07-55/`. Both `best_model.pt` checkpoints and `run.log`/`test_metrics.json` are saved under `save/` (gitignored). Treat it as equivalent in reliability to `train_gears.py` now — no longer an unverified port.

## Licensing checklist to keep in mind

Before publishing any derived outputs, GEARS/Norman/Adamson redistribution terms, scGPT and Geneformer weight licenses (Geneformer may be gated on Hugging Face), and scPerturb terms of use (if used) all need confirming — see `docs/BUILD_PLAN.md` §11. Ahlmann-Eltze et al. 2025 and the rebuttal preprint must be cited prominently since the entire project framing depends on transparently building on both.
