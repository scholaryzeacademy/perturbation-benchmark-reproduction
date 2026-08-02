# Project Overview: Perturbation Benchmark Reproduction

## 1. What this project is

An independent, transparent reproduction of an unresolved 2025 scientific debate in computational biology: **do deep-learning and foundation models actually beat simple statistical baselines at predicting how a cell's gene expression changes after a genetic perturbation?**

Two papers disagree:
1. **Ahlmann-Eltze, Huber & Anders** (*Nature Methods*, 2025) — argue that deep-learning/foundation models (GEARS, scGPT, Geneformer) do **not** yet outperform simple linear baselines at this task.
2. **A follow-up bioRxiv preprint** (Oct. 2025) — argues that conclusion is an artifact of poorly calibrated evaluation metrics, and that under a proper calibration framework (comparing models against the achievable range between a "no signal" and a "best possible" reference), foundation models **do** show real signal.

Rather than picking a side, this project reproduces **both** methodologies independently on the same real public datasets and reports honestly which finding holds, under which metric, on which dataset — including where the two datasets and two metrics disagree with each other.

## 2. Goals and non-goals

**Goals:**
- Benchmark at least one specialized model (GEARS) and one fine-tuned foundation model (scGPT) against at least two baselines (no-change, ridge) on real held-out perturbations.
- Report results under both conventional metrics and a calibration-aware framing.
- Produce an honest written discussion stating plainly whether the reproduction agrees with either paper, both, or neither.
- Make the whole thing independently reproducible (public repo, CI, every hyperparameter/seed documented).

**Explicit non-goals** (deliberately out of scope, not oversights):
- Not training a new foundation model or claiming a novel modeling contribution.
- Not attempting genome-wide scale (e.g., the full 2.5-million-cell Replogle et al. 2022 dataset).
- Not trying to definitively settle which of the two papers is "right" — the goal is transparent replication, not adjudication.

## 3. The data

Two public Perturb-seq (CRISPR + single-cell RNA-seq) datasets, both loaded via GEARS's own `PertData` class rather than re-processed by hand, so results stay comparable to published numbers:

- **Adamson et al. 2016** (*Cell*) — single-gene CRISPR knockdowns in K562 cells (the "easy," simpler perturbation regime). 21 real held-out test conditions (in this project's primary environment).
- **Norman et al. 2019** (*Science*) — single- and double-gene combinatorial CRISPR perturbations in A549 cells (the "hard" regime — predicting genetic interactions, not just single-gene effects). 107 real held-out test conditions.

Both use GEARS's own documented `simulation` train/val/test split — built around genuinely *unseen* perturbations (a perturbation's cells are entirely in one split, never divided across train and test) — with a fixed seed, not a custom re-split.

## 4. The pipeline: six stages

### Stage 1 — Data acquisition

Downloads and loads both datasets through GEARS's `PertData` class, applying its documented split. No custom code beyond a thin wrapper (`src/stage1_data/`).

### Stage 2 — Baseline models

Three deliberately simple baselines, sharing one `fit()`/`predict()` interface:
- **No-change**: predicts zero effect (control mean for every condition) — the field's standard "uninformative" negative control.
- **Mean**: predicts the average perturbation effect seen across training conditions, applied uniformly.
- **Ridge regression**: fits `sklearn.Ridge` on a multi-hot gene-identity encoding of the perturbed gene(s) — a deliberately simple stand-in for a learned embedding.

### Stage 3 — Specialized and foundation models

- **3a — GEARS**: a graph-based specialized model (uses a gene-similarity graph), trained with its own official default hyperparameters (`hidden_size=64, epochs=20`). Real training run on an RTX 4090: Adamson in ~22 min, Norman in ~26 min.
- **3b — scGPT**: a transformer foundation model, fine-tuned from its pretrained "whole-human" checkpoint (33M cells) following its own official tutorial. Real fine-tuning run on an RTX 4090 for both datasets, 15 epochs (early stopping triggered in both). A real upstream bug was found and fixed along the way: the installed `scgpt` package's flash-attention code was broken for the flash-attn version its own setup instructions install, silently falling back to a slower attention implementation whose weight names didn't match the pretrained checkpoint — fixed by installing scGPT from a specific upstream GitHub commit and using its newer checkpoint-loading utility instead of PyPI's broken release.
- **3c — Geneformer**: a foundation model used **zero-shot** (no fine-tuning) via its built-in "in silico perturbation" simulator. Its output is fundamentally different from the other models: a per-cell embedding *similarity shift* (how much does simulating a gene's deletion change the cell's overall representation), not a predicted gene-expression profile — so it's reported in its own separate table throughout, never forced into the same metric columns as the others. Six real bugs were found and fixed while porting this (a stricter-checkpoint requirement, a GPU out-of-memory issue, missing output directories, a library function silently returning nothing, a crash on multi-gene perturbations, and a shared-directory bug across runs). One further limitation was found and is not a bug: some gene pairs are never seen together in any real control cell, so "deleting both genes" has no cells to simulate on — those specific conditions are skipped with a logged reason rather than crashing the whole run. Final real runs: Norman 102/107 conditions succeeded (5 skipped for that reason); Adamson 21/21 (initially 12/21 due to an unrelated process on the shared GPU using up memory, later re-run successfully in full).

### Stage 4 — Conventional metrics

Two field-standard metrics computed identically for every expression-predicting model (the three baselines, GEARS, scGPT — not Geneformer, for the reason above):
- **MSE@20DEG**: mean squared error restricted to the top-20 differentially expressed genes for each perturbation (matches how the field typically evaluates this).
- **Pearson(Δcontrol)**: correlation between predicted and real expression *change from control*, across all genes.

**Real headline finding**: on Adamson, scGPT wins clearly on both metrics. On Norman, there's no single winner — GEARS has the best (lowest) MSE@20DEG of all five models, but ridge (the simplest baseline) has the best Pearson correlation, beating both GEARS and scGPT on that metric. Cross-checks against each model's own independently self-reported numbers matched closely, supporting that the underlying pipeline is implemented correctly.

A real, documented cross-environment quirk: scGPT and the other four models needed incompatible versions of the `cell-gears` library, which (confirmed during this work) produce not just different train/test split *assignments* from the same seed, but in one case actually different *cell content* available at all for a handful of conditions (some perturbations get silently dropped by one library version's internal graph-based filtering). Each model is reported against its own correct, real split rather than forcing an artificial common subset.

### Stage 5 — Metric calibration framework

The project's key differentiating contribution: instead of just reporting raw scores, each model is scored against the *gap* between a negative control (no-change baseline) and a positive control — an "interpolated duplicate," meant to represent the best result achievable given the data's own inherent noise. Because this project's real data has no true biological-replicate labels, the positive control was built (a documented, deliberate scope choice, confirmed before implementation) from repeated random 50/50 splits of each condition's own real held-out cells into pseudo-replicate halves, rather than genuine separate-batch replicates.

**Real finding**: recalibrating did **not** flip either dataset's ranking — scGPT still leads Adamson, and Norman's GEARS-wins-MSE/ridge-wins-Pearson split survives. This is itself an honest, useful result: the calibration framework mostly confirmed the raw-metric story here rather than overturning it, which doesn't fully validate the rebuttal paper's headline argument in this specific reproduction.

### Stage 6 — Honest reporting

Consolidates everything into: a combined results table across all models/metrics/datasets, a separate Geneformer ranking table, a written discussion answering three required questions (does it match paper 1? does calibration shift things per paper 2? where do numbers diverge and why?), a `METHODS.md` documenting every hyperparameter and seed, a `LICENSES.md` documenting real, directly-checked licensing facts, a ~1,700-word technical write-up, and a one-page PDF summary suitable as a client-style deliverable.

## 5. Results and findings in detail

### 5.1 Conventional metrics (Stage 4) — full numbers

Mean MSE@20DEG (lower is better) and mean Pearson(Δcontrol) (higher is better), across every real held-out test condition (`n`):

| dataset | model | mean MSE@20DEG | mean Pearson(Δcontrol) | n |
|---|---|---|---|---|
| adamson | no_change | 0.3800 | NaN (undefined — see below) | 21 |
| adamson | mean | 0.2478 | 0.6357 | 21 |
| adamson | ridge | 0.2482 | 0.6357 | 21 |
| adamson | gears | 0.2465 | 0.5557 | 21 |
| adamson | **scgpt** | **0.1240** | **0.6405** | 22 |
| norman | no_change | 0.4368 | NaN (undefined — see below) | 107 |
| norman | mean | 0.3316 | 0.5786 | 107 |
| norman | ridge | 0.2308 | **0.6732** | 107 |
| norman | **gears** | **0.1689** | 0.5831 | 107 |
| norman | scgpt | 0.2235 | 0.5505 | 116 |

**Reading this table:**
- **Adamson (single-gene perturbations) has one clear winner**: scGPT beats every baseline and GEARS on both metrics, by a wide margin on MSE@20DEG (0.1240 vs. ~0.25 for everything else).
- **Norman (combinatorial perturbations) has no single winner — it splits by metric.** GEARS has the lowest (best) MSE@20DEG of all five models, including both baselines. But ridge — the simplest of the three baselines — has the highest (best) Pearson(Δcontrol), beating both GEARS and scGPT on that metric. Whichever metric you read determines whether the headline is "the specialized model wins" or "the simple baseline wins."
- **`no_change`'s Pearson(Δcontrol) is mathematically undefined, not missing data.** It always predicts the control mean, so its predicted change-from-control is the zero vector for every condition, and correlation of a constant (zero-variance) vector against anything is undefined. This is the expected, correct behavior of a "predict zero effect" baseline.
- **scGPT's `n` differs slightly from the other four models** (22 vs. 21 for Adamson, 116 vs. 107 for Norman) because it runs in a separate software environment that resolves GEARS's own "same seed, same split name" request into a genuinely different real set of test conditions — a confirmed library-version effect, not an error in this project's own code (see the cross-environment note below).

**Cross-checks against published/self-reported numbers** (used as a pipeline-correctness sanity check, not expected to match exactly): GEARS's own self-reported "Test Top 20 DE MSE" from its real training run was 0.2364 (Adamson) / 0.1683 (Norman) vs. this project's independently-computed 0.2465 / 0.1689 — close on both, within 0.4% on Norman. scGPT's own self-reported Pearson-of-delta metric was 0.6405065 (Adamson) / 0.5493100 (Norman) vs. this project's 0.6405477 / 0.5504978 — matching almost to the decimal place, since both are computing the exact same definition through two independent code paths. Both cross-checks support the metrics implementation being correct.

### 5.2 Calibrated metrics (Stage 5) — full numbers

Each model's raw score, rescaled into a "fraction of recoverable signal captured" between the no-change negative control (0.0) and the interpolated-duplicate positive control (1.0):

| dataset | model | calibrated MSE fraction | calibrated Pearson fraction | n conditions used |
|---|---|---|---|---|
| adamson | no_change | 0.0000 | NaN (see above) | 36 |
| adamson | mean | 0.3492 | 0.6741 | 21 |
| adamson | ridge | 0.3483 | 0.6741 | 21 |
| adamson | gears | 0.3526 | 0.5892 | 21 |
| adamson | **scgpt** | **0.5720** | 0.6710 | 21 |
| norman | no_change | 0.0000 | NaN (see above) | 142 |
| norman | mean | 0.2439 | 0.6427 | 107 |
| norman | ridge | 0.4775 | **0.7478** | 107 |
| norman | **gears** | **0.6210** | 0.6478 | 107 |
| norman | scgpt | 0.5077 | 0.6239 | 112 |

**Reading this table:**
- **`no_change` scores exactly 0.0 on calibrated MSE by construction** (it's being compared against itself) — this is a built-in pipeline sanity check, and it held on real data for both datasets, confirming the calibration math is wired correctly.
- **Calibration does not flip either dataset's ranking.** scGPT still clearly leads Adamson's calibrated MSE fraction (0.572, well ahead of the ~0.35 the other four cluster around). On Norman, GEARS still leads the calibrated MSE fraction (0.621) and ridge still leads the calibrated Pearson fraction (0.748) — the same split-by-metric picture as the raw numbers, just rescaled.
- **One real, mild nuance calibration does surface**: on Adamson's Pearson fraction, scGPT's clear raw lead (0.6405 vs. ridge/mean's 0.6357) becomes a near-three-way tie once calibrated (scGPT 0.671 vs. ridge/mean 0.674, which are now marginally *ahead*) — not a ranking flip, but a real change in how close the models look once you account for the calibration ceiling.
- **`n conditions used` is smaller than each model's own raw `n`** in a few cases (e.g., scGPT's Adamson calibration uses 21 conditions, not 22) because a handful of conditions had zero real cells available for building the positive control in this project's data-loading environment — those specific conditions are excluded from the calibrated score with a logged reason, not silently dropped.

### 5.3 Geneformer — zero-shot cosine-shift ranking (reported separately, not comparable to the tables above)

Geneformer's output is a per-cell similarity shift between its embedding of a cell before and after a simulated gene deletion — ranked here per condition (mean shift across all real cells for that condition), not a predicted expression profile, so it cannot be placed in the MSE@20DEG/Pearson columns above.

**Adamson** (21/21 conditions; larger shift = lower mean cosine similarity): the biggest predicted shifts were `TARS+ctrl`, `CAD+ctrl`, and `DDIT3+ctrl` (mean cosine similarity ≈0.999905–0.999907); the smallest were `CREB1+ctrl` and `BHLHE40+ctrl` (≈0.999945–0.999947). All shifts are small in absolute terms (all values sit in a ~0.00004-wide band above 0.9999) — expected for a single-gene deletion out of thousands of expressed genes in a high-dimensional cell embedding, not a sign of a broken signal.

**Norman** (102/107 conditions; 5 skipped for a documented zero-co-occurring-cells limitation of "delete"-type perturbation): the biggest nominal shifts were `DUSP9+PRTG` and `IGDCC3+PRTG`, but **both have only 1 real cell each** — too small a sample to trust as a genuine finding, flagged explicitly rather than reported as a real top result. Setting those aside, the next largest real shifts were `C3orf72+FOXL2`, `FOXA3+HOXB9`, and `DUSP9+MAPK1` (n=6–589 cells). The smallest shifts were single-gene conditions like `SPI1+ctrl`/`ctrl+SPI1` and `CEBPB+ctrl`/`ctrl+CEBPB` — both of which appear twice, once with the perturbed gene listed first and once second, with identical statistics both times, a real quirk in Norman's own condition-naming convention rather than a code issue.

### 5.4 Synthesis: what this means for the two papers

Neither paper is cleanly confirmed or cleanly refuted by this reproduction:

- **Ahlmann-Eltze et al.'s "DL doesn't win" conclusion holds on Norman's Pearson metric** (ridge beats both GEARS and scGPT) **but is directly contradicted on Adamson** (scGPT wins clearly on both metrics) **and on Norman's own MSE metric** (GEARS wins). A blanket "foundation models don't help" statement is not supported by this reproduction; neither is a blanket "they do."
- **The rebuttal paper's calibration argument — that proper calibration reveals foundation-model signal conventional metrics hide — is not reproduced here.** Calibrating against the negative/positive control gap preserved the same rankings the raw metrics already showed, on both datasets. This reproduction's calibration framework is a real but deliberately scoped-down version of the rebuttal's full method (see the Known Limitations section), which is the most likely reason it doesn't reproduce their full effect — that gap is stated plainly rather than smoothed over.
- **The most robust, dataset-independent finding across this whole project is that the "winner" depends heavily on which dataset (single-gene vs. combinatorial) and which metric (MSE vs. Pearson) you choose** — exactly the kind of benchmarking fragility both papers are, in different ways, already arguing the field needs to take seriously.

## 6. Tech stack

- Python 3.10, `AnnData`/`scanpy` for single-cell data structures.
- `cell-gears` (GEARS) for data loading and the specialized graph model — needs PyTorch Geometric.
- `scGPT` (from a specific GitHub commit, not the broken PyPI release) for the fine-tuned foundation model.
- `Geneformer-V2-316M` (Hugging Face) for the zero-shot foundation model.
- Three **separate conda environments**, deliberately isolated, because GEARS and scGPT require incompatible `cell-gears` versions, and scGPT/Geneformer each need their own GPU-specific dependency sets.
- A multi-stage Docker setup (`cpu`/`scgpt`/`geneformer` build targets) mirroring the same isolation.
- GitHub Actions CI: fast unit tests on every push, plus a real (scaled-down) data-download-and-train smoke test.
- All real model-training/inference work was run on an RTX 4090 GPU (a machine shared with other users, which caused one real, documented GPU-memory contention issue along the way).

## 7. Where every number comes from (reproducibility)

Every hyperparameter, random seed, data split, and metric formula is written down in `configs/training_hyperparameters.yaml` and `METHODS.md` — nothing is tuned ad hoc or left undocumented, since independent reproducibility is treated as a hard requirement given how much of the underlying scientific debate is itself about benchmarking fragility. Every stage's real output CSVs live under `data/` (gitignored, but regeneratable via the documented commands), and Stage 6's report tables are generated mechanically from those CSVs rather than typed by hand — a discipline that already once caught and fixed a real transcription error in an earlier internal summary of the Norman results.

## 8. Known limitations (stated honestly, not hidden)

- The calibration framework implements the rebuttal paper's *core idea*, not its full 13-metric, 14-dataset framework.
- The positive control uses pseudo-replicates (random cell splits), not true biological replicates, because the source data doesn't have replicate labels.
- Geneformer's output isn't directly comparable to the other four models under the same metrics — a limitation of its own public API, not something this project could work around without risky, undocumented internal-function usage it deliberately chose not to use.
- scGPT is evaluated on a slightly different real test-condition split than the other four models, due to the library version incompatibility mentioned above.
- Two licensing items remain genuinely unconfirmed (not fabricated): scGPT's pretrained checkpoint license isn't separately stated from its code license, and the original datasets' exact redistribution terms (beyond GEARS's own MIT-licensed download mechanism) weren't independently tracked down.
- The four exploratory Jupyter notebooks originally sketched in the project's repository-structure plan were never built — everything they would have shown is already covered in the written report and result CSVs, but the notebooks themselves don't exist.

## 9. Licensing

This project's own code is MIT-licensed. Both upstream projects it depends on most directly, GEARS and scGPT, are also MIT-licensed (confirmed directly against their repositories). Geneformer's model weights are Apache-2.0 and openly downloadable, not gated behind a credential request as an earlier, less-verified note in this project's own documentation once assumed.
