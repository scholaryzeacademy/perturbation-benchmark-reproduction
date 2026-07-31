# METHODS

Every hyperparameter, random seed, data split, and metric definition used in this reproduction, organized by pipeline stage, so an independent party can reproduce every reported number without reading source code. All values below are also machine-readable in `configs/training_hyperparameters.yaml`, which is the actual source of truth this document summarizes — if the two ever disagree, trust the config file.

## Stage 1 — Data acquisition

- **Source**: Norman et al. 2019 (*Science*) and Adamson et al. 2016 (*Cell*) Perturb-seq datasets, downloaded via GEARS's own `PertData` class (`gears.PertData`), not re-hosted or re-processed by this project.
- **Split**: GEARS's own documented `simulation` split (built around *unseen* perturbations — a perturbation's cells are entirely in train, val, or test, never split across them), `seed=1`, `train_gene_set_size=0.75`. Used as-is rather than re-splitting, to preserve comparability with published numbers.
- **Dataloader parameters**: `batch_size=32`, `test_batch_size=128` (only relevant to GEARS's own training loop, Stage 3a).
- **Real split sizes actually produced** (differ by `cell-gears` version — see the cross-version note under Stage 4/5 below): under `cell-gears==0.1.2` (this project's `perturb-bench-gears-gpu` env), Adamson's test set is 21 conditions (all `unseen_single`), Norman's is 107 (36 `unseen_single`, 9 `combo_seen0`, 43 `combo_seen1`, 19 `combo_seen2`). Under `cell-gears==0.0.2` (`perturb-bench-scgpt` env), Adamson is 22 conditions, Norman is 116 — a real, confirmed difference from the same seed and split name.

## Stage 2 — Baseline models

Shared contract: `fit(adata, train_conditions) -> self`, `predict(condition) -> np.ndarray` (an `n_genes`-long mean expression profile).

- **No-change**: predicts the real control-cell mean for every condition (ignores `train_conditions`).
- **Mean**: predicts control mean + the mean of (perturbed-condition mean − control mean) across every training condition.
- **Ridge**: `sklearn.linear_model.Ridge(alpha=1.0, random_state=1)`, fit to predict each training condition's expression delta from control, using a multi-hot gene-identity encoding of the perturbed gene(s) as the input feature (a deliberate simplification of an ESM-2-style learned embedding, documented as a scope choice in `ridge_baseline.py`'s docstring). A test condition whose gene(s) never appear in training encodes as all-zero, so the model falls back to its intercept — expected, not a bug.

## Stage 3a — GEARS

GEARS's own documented library defaults, not tuned by this project: `hidden_size=64`, `epochs=20`, `lr=0.001`, `weight_decay=0.0005`. Real training run (RTX 4090): Adamson 1309.5s, Norman 1541.6s. Adamson `Test Top 20 DE MSE: 0.2364`; Norman `Test Top 20 DE MSE: 0.1683` (this project's own cross-check anchor: a commonly cited published Norman MSE@20DEG figure is ~0.176).

## Stage 3b — scGPT

Fine-tuned from the `whole-human` pretrained checkpoint per `bowang-lab/scGPT`'s own `Tutorial_Perturbation.ipynb`. Key hyperparameters (full list in `configs/training_hyperparameters.yaml`'s `stage3b_scgpt`): `lr=0.0001`, `batch_size=64`, `epochs=15` (config max; early stopping triggered in both real runs), `early_stop=10`, `dropout=0`, `amp=true`, `seed=42`. Architecture params (`embsize`/`d_hid`/`nlayers`/`nhead`/`n_layers_cls`) are overridden by the pretrained checkpoint's own `args.json` at load time and only matter if training from scratch. Real runs (RTX 4090): Adamson best val Pearson 0.9906 (epoch 14); Norman best val Pearson 0.9909 (epoch 14).

## Stage 3c — Geneformer

Zero-shot in silico perturbation (`Geneformer-V2-316M`, no fine-tuning) — outputs a per-cell cosine-similarity shift between pre/post-perturbation embeddings, not a per-gene expression prediction (see the Metrics section below for why this means Geneformer is reported in a separate table, not the same MSE@20DEG/Pearson columns). Key parameters: `perturb_type=delete`, `emb_mode=cls`, `forward_batch_size=16` (sized for a 24GB card), `stats_mode=aggregate_data`, `seed=42`. `n_counts` for tokenization is reconstructed via `expm1(X).sum(axis=1)` since the source data only ships post-normalization expression, not raw counts — a documented approximation (preserves each cell's within-cell gene rank order, the signal the tokenizer's rank-value encoding actually uses).

Real production runs (RTX 4090, `max_ncells=200`): Norman 102/107 conditions succeeded, 5 skipped for a genuine zero-co-occurring-cells limitation of "delete"-type perturbation (some gene pairs are never co-tokenized in any real control cell). Adamson 21/21 (initially 12/21 on GPU-memory contention from an unrelated process; the remaining 9 succeeded on re-run once that process freed memory).

## Stage 4 — Conventional metrics

Implemented in `src/stage4_conventional_metrics/metrics.py`, applied identically to every model via one `evaluate_model(predict_fn, adata, test_conditions, model_name)` entry point.

- **MSE@20DEG**: mean squared error between predicted and real mean post-perturbation expression, restricted to the top-20 differentially expressed genes for that condition vs. control. DE genes selected via `scanpy.tl.rank_genes_groups` (t-test, `rankby_abs=True`) — matches GEARS's own `rank_genes_groups_by_cov` convention, but does **not** additionally filter to GEARS's own "non-dropout" gene subset, so a modest difference from GEARS's self-reported number is expected, not a bug.
- **Pearson(Δcontrol)**: Pearson correlation between predicted and real expression *change from control* (not raw expression), across all genes (not just the top-20 DE set).
- **`no_change`'s Pearson(Δcontrol) is mathematically NaN by construction**: its predicted delta is always the zero vector, and correlation of a zero-variance vector against anything is undefined.
- **Real cross-env split mismatch**: baselines/GEARS (`cell-gears==0.1.2`) and scGPT (`cell-gears==0.0.2`) are evaluated on genuinely different real test-condition sets from the same seed (see Stage 1 above) — each model is reported against its own real, correct split rather than an intersection-only subset.
- **Real cross-env cell-content difference** (found during Stage 5 work, deeper than the split-assignment note above): `cell-gears==0.1.2`'s `PertData.load()` drops conditions "not in the GO graph" from `pert_data.adata` entirely (confirmed: 65899 vs. 68603 total Adamson cells vs. the raw h5ad) — `cell-gears==0.0.2` does not do this filtering, so a handful of conditions (e.g. `AMIGO3+ctrl`) appear with real scGPT scores but zero cells under the other env.

## Stage 5 — Metric calibration

Implemented in `src/stage5_calibration/calibration.py`. Compares each model's Stage 4 score against the gap between the no-change negative control and an "interpolated duplicate" positive control, producing a calibrated fraction of recoverable signal captured (1.0 = matches the positive-control ceiling, 0.0 = matches no-change).

- **Positive control**: since this project's real data has no true biological-replicate/batch column (confirmed: `adata.obs` only has `condition`/`cell_type`/`dose_val`/`control`/`condition_name`), it is built from `n_splits=10` random 50/50 splits of each real test condition's own held-out cells into two pseudo-replicate halves (`seed=1`), each split scored in both directions (20 samples per condition, averaged) using the exact same `mse_at_20deg`/`pearson_delta_control` functions models are scored with.
- **Negative-control Pearson reference**: since `no_change`'s own Pearson is NaN (see Stage 4), `0.0` is substituted as the "no linear relationship" reference (`pearson_negative_control_reference` in the config) — a documented substitution.
- **Aggregation**: pooled, not per-condition-ratio-then-averaged — `calibrated_mse_fraction = (mean_mse_negctrl − mean_mse_model) / (mean_mse_negctrl − mean_mse_posctrl)`, `calibrated_pearson_fraction = (mean_pearson_model − 0.0) / (mean_pearson_posctrl − 0.0)`, each computed once per (dataset, model) from three already-averaged numbers. Avoids a per-condition ratio blowing up when one weak-true-effect condition's positive control barely beats no-change for that condition alone. `no_change`'s own `calibrated_mse_fraction` is exactly `0.0` by construction — a pipeline sanity check, confirmed on real data for both datasets.
- **Negative control recomputed fresh** per dataset over the union of every condition any model was scored on (not read from Stage 4's own `no_change` CSV), since Stage 4's `no_change` CSV only covers baselines/GEARS's own split and is missing the few conditions unique to scGPT's split (see Stage 4's cross-env note).

## Stage 6 — Reporting

`src/stage6_reporting/build_report_tables.py` generates the results tables embedded in the technical write-up directly from Stage 4/5's summary CSVs and Stage 3c's Geneformer output — not hand-transcribed, to avoid the transcription-error risk this project already hit once (see `CLAUDE.md`'s Stage 4/5 Working Conventions entries and `reports/final_writeup/technical_writeup.md` for the corrected finding).
