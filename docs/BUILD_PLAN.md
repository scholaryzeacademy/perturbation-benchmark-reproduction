# BUILD_PLAN.md
## Single-Cell Perturbation Prediction: Foundation Models vs. Simple Baselines
### A Rigorous, Honest Reproduction — And Extension — of the 2025 Benchmarking Controversy

---

> **Project type:** Flagship open-source portfolio piece (Single-Cell & Spatial Omics vertical)
> **Status:** Planning
> **License target:** MIT/Apache-2.0 for original code; third-party model/dataset licenses reviewed individually (see §11)

---

## 0. One-Paragraph Summary

We reproduce, on public Perturb-seq data, the now-well-known 2025 finding that deep-learning single-cell foundation models (scGPT, Geneformer) and specialized perturbation-prediction models (GEARS) fail to beat deliberately simple baselines (a "no change" control and a mean/linear baseline) at predicting post-perturbation gene expression [Ahlmann-Eltze, Huber & Anders, *Nature Methods*, 2025]. Critically, **we do not stop there** — a direct follow-up preprint published two months later argues the original "linear beats DL" conclusion is itself an artifact of poorly calibrated evaluation metrics, and proposes a metric-calibration framework (positive/negative controls, a "dynamic range fraction" measure, rank-based metrics) under which the deep learning models *do* show real signal [bioRxiv, October 2025]. Our project implements **both sides of this live, unresolved scientific debate** on the same dataset and reports honestly which conclusion holds under which metric — this dual framing, not just a single reproduction, is the project's actual differentiator and the clearest possible demonstration of the "rigorous and honest benchmarking" positioning this whole portfolio strategy is built on.

---

## 1. Goals & Non-Goals

### 1.1 Goals
- Reproduce the core empirical comparison (foundation models / specialized DL models vs. simple baselines) on at least one public, standard Perturb-seq dataset, using tools and access paths verified to actually work.
- Implement the metric-calibration critique from the follow-up paper well enough to show, concretely, whether the "DL doesn't help" conclusion survives under better-calibrated metrics on our chosen dataset.
- Report results with full honesty, including the genuinely uncomfortable possibility that our own reproduction doesn't cleanly match either published paper's conclusion — that outcome is still a legitimate, useful finding worth publishing as-is.
- Produce a reusable, documented benchmarking harness that a client could point at their own perturbation dataset.

### 1.2 Non-Goals
- We are **not** training a new foundation model or claiming a novel modeling contribution — this is a benchmarking and evaluation-methodology project.
- We are **not** attempting genome-wide scale (e.g., full Replogle et al. 2022 2.5M-cell dataset) in v1 — dataset choice is deliberately scoped for tractable compute (see §4, §10).
- We are **not** trying to definitively resolve the Ahlmann-Eltze vs. rebuttal debate — that's an open scientific question involving groups with far more resources than this project; our contribution is an independent, transparent replication on a dataset/setup we fully control and document.

---

## 2. Definition of Done

1. At least one foundation model (scGPT, fine-tuned) and one specialized perturbation model (GEARS) are benchmarked against at least two baselines (a "no-change" control and a linear/ridge regression baseline) on unseen held-out perturbations from a standard public dataset.
2. Results are reported under **both** conventional metrics (MSE on top-20 differentially expressed genes, Pearson correlation of the delta from control) **and** at least one calibration-aware approach inspired by the rebuttal paper (minimum: implement the "no-change" and a proper positive-control baseline side by side, and report a calibration-sensitive framing of the ranking — see §6, Stage 5, for the honest scope of what's achievable without the original paper's full codebase).
3. A written, honest discussion section explicitly states whether our results align with Ahlmann-Eltze et al. 2025, the rebuttal, both, or neither — with the actual numbers shown, not just a qualitative claim.
4. Public GitHub repo, documented, with CI smoke test.
5. Technical write-up (~1,500–2,500 words) ready for publication.

---

## 3. Architecture Overview

```
                    ┌───────────────────────────┐
                    │ STAGE 1: Data Acquisition   │
                    │ (GEARS PertData: Norman +   │
                    │ Adamson, standard splits)   │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ STAGE 2: Baseline Models    │
                    │ (no-change, mean, ridge     │
                    │ regression on embeddings)   │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
    ┌─────────▼────────┐ ┌────────▼─────────┐ ┌───────▼────────┐
    │ STAGE 3a: GEARS    │ │ STAGE 3b: scGPT   │ │ STAGE 3c:       │
    │ (official, graph-  │ │ (fine-tuned for   │ │ Geneformer      │
    │ based specialist   │ │ perturbation)     │ │ (zero-shot in   │
    │ model)              │ │                   │ │ silico perturb) │
    └─────────┬────────┘ └────────┬─────────┘ └───────┬────────┘
              │                   │                   │
              └───────────────────┼───────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ STAGE 4: Conventional       │
                    │ Metrics (MSE@20DEG,         │
                    │ Pearson delta)              │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ STAGE 5: Metric Calibration │
                    │ Framework (positive control,│
                    │ dynamic-range-aware ranking)│
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │ STAGE 6: Honest Reporting   │
                    │ (both metric regimes side   │
                    │ by side, explicit discussion)│
                    └────────────────────────────┘
```

---

## 4. Data Sources (Verified)

| Dataset | Role | Access | Notes |
|---|---|---|---|
| **Norman et al. 2019** | Primary benchmark dataset | Auto-downloaded via the `gears` Python package's `PertData` class (`pert_data = PertData('./data'); pert_data.load(data_name='norman')`) | K562 cells, CRISPRa; includes both single-gene and dual-gene (combinatorial) perturbations — this combinatorial structure is exactly what makes it the standard, most-cited benchmark for testing whether a model captures non-additive genetic interactions, and it's the dataset both the original benchmark and much of the DL-vs-baseline literature center on. GEARS's own documented train/val/test split (unseen single/double perturbations) is used as-is rather than re-splitting — this preserves comparability with published numbers. |
| **Adamson et al. 2016** | Secondary/simpler validation dataset | Same `PertData` mechanism, `data_name='adamson'` | K562 cells, CRISPRi, single-gene perturbations only (no combinatorial complexity) — a useful simpler test case to check whether conclusions differ between the "easy" (single-perturbation) and "hard" (combinatorial) regime. |
| **scPerturb** | Reference/cross-check, optional extension | `scperturb.org` — harmonized AnnData-format repository of 44 datasets from 25 publications [Peidli et al., *Nature Methods*, 2024] | Useful if we want a third, independently-processed dataset for a robustness check in a v2 extension; not required for v1's Definition of Done. |

**Action item before build starts:** confirm current GEARS package version and dataset download URLs are still live (these occasionally move between Harvard Dataverse mirrors) — do this as the very first task, since a broken data-download step blocks everything downstream.

---

## 5. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.10+ | Standard across the whole stack |
| Environment management | Conda + `environment.yml`; separate Docker image for GPU stages | scGPT's flash-attention dependency is CUDA-version-sensitive; isolate it |
| Data structures | `AnnData` / `scanpy` | The field standard; GEARS, scGPT, and Geneformer all consume or produce AnnData-compatible formats |
| Specialized perturbation model | `GEARS` — `pip install cell-gears` (package name on PyPI; verify exact current name at build time), from `github.com/snap-stanford/GEARS` | Confirmed open source, auto-downloads Norman/Adamson/Dixit via its own `PertData` class — this is the standard non-foundation-model DL baseline used across nearly every paper in this space |
| Foundation model 1 | `scGPT` — `pip install scgpt` (optional `flash-attn` for speed), from `github.com/bowang-lab/scGPT` | Has documented fine-tuning examples specifically for perturbation prediction; this is one of the two foundation models most directly implicated in the original benchmarking controversy |
| Foundation model 2 | `Geneformer` — `git lfs install && git clone https://huggingface.co/ctheodoris/Geneformer && pip install .` | Supports zero-shot "in silico perturbation" directly, which is a meaningfully different modeling approach from scGPT's fine-tuning route — including both broadens the reproduction beyond a single model family |
| Baseline models | Custom: `scikit-learn` `Ridge` regression on top of either raw expression or ESM-2-style embeddings (matching the "ridge regression to predict gene expression... from perturbed embeddings" approach documented in Ahlmann-Eltze et al.'s own methods) | Deliberately simple, deliberately well-documented — the entire point is that these are *not* sophisticated models |
| Metrics — conventional | Custom implementation of MSE@20DEG and Pearson(Δcontrol), using `scanpy.tl.rank_genes_groups` for DEG identification (t-test method, matching the field-standard approach documented in multiple benchmarking papers) | Matches the exact metric definitions used across the papers we're reproducing, for direct comparability |
| Metrics — calibration framework | Custom reimplementation, from the published methodology description, of: (a) a "no-change" negative control, (b) an "interpolated duplicate" positive control (a synthetic upper-bound baseline constructed by interpolating between two real replicate profiles), and (c) a **dynamic range fraction** calibration score comparing each model's performance to the gap between the negative and positive controls | Directly reproducing the rebuttal paper's proposed fix — see §6 Stage 5 for honest scope caveats |
| Compute | GPU required for scGPT/Geneformer fine-tuning stages (a single mid-range GPU, e.g., a T4/A10-class cloud instance, is sufficient at this dataset scale based on published benchmark setups); GEARS and baseline stages are CPU-feasible | Budget cloud GPU hours explicitly — see §10 |
| CI | GitHub Actions, smoke-testing the baseline + GEARS pipeline on a small perturbation subset (foundation-model fine-tuning is too heavy for CI; run those manually/on-demand) | Keeps CI fast and free-tier-compatible |

---

## 6. Pipeline Stage Detail

### Stage 1 — Data Acquisition
- Use GEARS's `PertData` class to download and load Norman and Adamson, preserving the package's own documented train/val/test split (this split is itself constructed around *unseen* perturbations, which is directly relevant to the generalization theme running through this whole project series)
- **Owner:** Bioinformatics / Data Engineer

### Stage 2 — Baseline Models
Three baselines, each deliberately simple and fully documented:
1. **No-change baseline:** predict that perturbed-cell expression equals control-cell expression (i.e., predict zero effect). This is the "uninformative" negative-control baseline used across the literature.
2. **Mean baseline:** predict the mean observed perturbation effect across the training set, applied uniformly to test perturbations.
3. **Ridge regression baseline:** following the Ahlmann-Eltze et al. methodology, fit ridge regression using available gene/perturbation embeddings to predict expression change, then apply to held-out perturbations.
- **Owner:** Biostatistics + AI Engineer

### Stage 3a — GEARS (Specialized Graph-Based Model)
- Train GEARS using its own documented training procedure and default hyperparameters on the Norman/Adamson training splits
- **Owner:** AI Engineer

### Stage 3b — scGPT (Foundation Model, Fine-Tuned)
- Follow scGPT's documented perturbation-prediction fine-tuning example, fine-tuning on the same training split as GEARS for direct comparability
- **Owner:** AI Engineer (GPU-stage owner)

### Stage 3c — Geneformer (Foundation Model, Zero-Shot In Silico Perturbation)
- Use Geneformer's built-in in silico perturbation functionality in zero-shot mode (no task-specific fine-tuning), since this is one of its headline documented capabilities and a meaningfully different evaluation regime from scGPT's fine-tuned approach
- **Owner:** AI Engineer

### Stage 4 — Conventional Metrics
- Compute MSE@20DEG and Pearson(Δcontrol) for every model (including all three baselines) on the held-out test perturbations, exactly matching field-standard metric definitions so our numbers are comparable to published results
- **Owner:** Biostatistics

### Stage 5 — Metric Calibration Framework (Honest Scope)
This is the project's key differentiating contribution, and also where we need to be most careful about honestly representing what we can and can't fully reproduce:
- **What we can build directly from the published methodology:** the negative control (no-change baseline, already built in Stage 2), and a basic positive-control comparison using held-out biological replicates (an "interpolated duplicate" — averaging two real replicate profiles as an upper-bound reference point for how much signal is recoverable at all in this data).
- **What requires explicit honesty about scope:** the rebuttal paper's full "dynamic range fraction" metric and its complete set of 13 evaluated metrics across 14 datasets represent a larger evaluation-methodology research effort than this project can fully replicate in scope. Our approach: implement the **core idea** (comparing each model's performance against the gap between negative and positive controls, rather than reporting raw MSE/correlation in isolation) rather than claiming a byte-for-byte reproduction of their complete statistical framework. **State this scope limitation explicitly in the write-up** — this is exactly the kind of honest scoping this whole project is meant to model, not a shortcut to hide.
- **Owner:** Biostatistics (this stage should have explicit statistical sign-off given its centrality to the project's credibility)

### Stage 6 — Honest Reporting
- Present results as a table: each model (including baselines) × each metric (conventional and calibration-aware), on both datasets
- Write an explicit discussion section addressing: does our reproduction match Ahlmann-Eltze et al.'s conclusion under conventional metrics? Does recalibrating shift the ranking, consistent with the rebuttal's argument? Where do our specific numbers diverge from either paper's published numbers, and what's the most likely explanation (dataset subset differences, hyperparameter differences, compute-budget differences)?
- **Owner:** Whole team reviews before publication; Biostatistics has final sign-off on any statistical claim

---

## 7. Repository Structure

```
perturbation-benchmark-reproduction/
├── README.md
├── METHODS.md
├── LICENSES.md
├── environment.yml
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml
├── src/
│   ├── stage1_data/
│   ├── stage2_baselines/
│   │   ├── no_change.py
│   │   ├── mean_baseline.py
│   │   └── ridge_baseline.py
│   ├── stage3_models/
│   │   ├── gears_wrapper.py
│   │   ├── scgpt_wrapper.py
│   │   └── geneformer_wrapper.py
│   ├── stage4_conventional_metrics/
│   ├── stage5_calibration/
│   └── stage6_reporting/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_vs_models_norman.ipynb
│   ├── 03_baseline_vs_models_adamson.ipynb
│   └── 04_calibration_analysis.ipynb
├── configs/
│   └── training_hyperparameters.yaml   # documented explicitly for reproducibility
├── tests/
└── reports/
    └── final_writeup/
```

---

## 8. Milestones & Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| **Phase 0 — Setup & Diligence** | Week 1 | Data download confirmed working end-to-end for both datasets; GPU cloud environment provisioned and cost-estimated; license review (§11) started |
| **Phase 1 — Baselines** | Week 2 | Stage 2 complete on both datasets; sanity-checked against GEARS's own documented no-perturb baseline numbers where published |
| **Phase 2 — Specialized & Foundation Models** | Weeks 3–5 | Stages 3a/3b/3c trained and generating predictions on held-out test perturbations; this is the most compute- and debugging-intensive phase — budget extra buffer time here specifically |
| **Phase 3 — Conventional Metrics** | Week 5–6 | Stage 4 complete; results table produced; first-pass comparison against published numbers |
| **Phase 4 — Calibration Framework** | Weeks 6–7 | Stage 5 complete, with explicit scope documentation per the honesty caveat above |
| **Phase 5 — Reporting & Publication** | Weeks 7–9 | Stage 6 complete; technical write-up drafted, reviewed, and published; repo finalized and made public |

**Total: ~9 weeks** — one week longer than the antibody developability project, reflecting the heavier compute/debugging burden of fine-tuning two separate foundation model architectures rather than calling well-packaged structure-prediction tools.

---

## 9. Validation & QA Strategy

1. **Reproduce, don't just cite, the "no-change baseline is competitive" finding first**, before building anything more complex — if our own no-change baseline doesn't land in a plausible range relative to published numbers, debug the pipeline before proceeding, since this is the paper's own headline empirical anchor point.
2. **Cross-check GEARS numbers against its own published benchmark results** on Norman (published MSE@20DEG figures exist in multiple papers, e.g., ~0.176 in one commonly cited comparison) as a pipeline-correctness sanity check before trusting our scGPT/Geneformer runs.
3. **Explicitly separate "our numbers don't match the paper's numbers" (a debugging problem) from "our numbers show a genuinely different pattern" (a real finding)** — document the investigation either way rather than silently adjusting methodology until numbers look "right."
4. **Run both datasets (Norman combinatorial, Adamson single-gene) and report both** — if conclusions differ between the "easy" and "hard" perturbation regime, that's a genuinely interesting, honest, publishable finding in its own right, consistent with how the field's own literature already frames combinatorial prediction as the harder, more diagnostic test.
5. **Document every hyperparameter and random seed** in `configs/` — given how much current single-cell benchmarking literature (Section 5/§5 of the earlier skill-stack document) is explicitly about benchmarking fragility, our own reproduction needs to be maximally reproducible by someone else, or it undermines the exact credibility this project is meant to build.

---

## 10. Compute & Infrastructure Requirements

This project is meaningfully heavier on compute than the antibody developability pipeline — plan accordingly:
- **GPU required** for scGPT fine-tuning and, to a lesser extent, Geneformer in silico perturbation at reasonable speed. A single cloud GPU instance (e.g., an A10/T4-class instance, available on-demand from AWS/GCP/Azure) is adequate at the Norman/Adamson dataset scale (tens of thousands of cells, not millions) — this is not a large training cluster project.
- **Estimated cloud cost:** budget for on the order of 20–40 GPU-hours across Phase 2's model-training/fine-tuning work, plus buffer for debugging re-runs — DevOps/cloud team member should get a firm cost estimate during Phase 0 before the team commits to the full timeline.
- **Storage:** Perturb-seq AnnData objects at this scale are modest (low GBs, not TBs) — no special infrastructure needed beyond standard cloud storage.
- **DevOps/cloud team member's core contribution:** provisioning and tearing down GPU instances cost-efficiently (spot/preemptible instances are a good fit for the training-heavy Phase 2), plus the same Docker/CI reproducibility work as the antibody project.

---

## 11. Licensing & Compliance Checklist

- [ ] Confirm GEARS package license (check repository LICENSE file directly) and dataset redistribution terms for Norman/Adamson as distributed through GEARS's own download mechanism
- [ ] Confirm scGPT and Geneformer model weight license terms — Geneformer in particular is distributed via a gated/credentialed Hugging Face repository in some versions; confirm current access and redistribution terms before publishing any derived outputs
- [ ] Confirm scPerturb terms of use if used for the optional v2 cross-check extension
- [ ] Cite Ahlmann-Eltze, Huber & Anders (2025) and the follow-up calibration preprint clearly and prominently — this project's entire framing depends on transparently building on both, not presenting either as our own original finding
- [ ] Choose and apply an explicit open-source license for original code (MIT or Apache-2.0)

---

## 12. Team & Role Allocation

| Role | Primary Responsibility |
|---|---|
| **AI Engineer** | Stages 3a/3b/3c (model training/fine-tuning) — the most technically demanding stages |
| **Bioinformatics** | Stage 1 (data), sanity-checking biological plausibility of DEG selection and dataset splits |
| **Biostatistics** | Stages 2, 4, 5 — baseline design, metric implementation, and final statistical sign-off on the calibration framework and all claims in the write-up |
| **Software Engineer** | Repository architecture, Stage 6 reporting, reproducibility tooling |
| **DevOps/Cloud** | GPU provisioning/cost management (§10), CI, Docker reproducibility |
| **Genetics/Pharma domain expert** | Plain-language interpretation of what the combinatorial vs. single-gene distinction actually means biologically, for the non-computational sections of the write-up |
| **Project owner** | Owns Definition of Done and the Phase 2 buffer-time decision if compute/debugging runs over schedule |

---

## 13. Deliverables Checklist

- [ ] Public GitHub repository with complete documentation
- [ ] `METHODS.md` with every hyperparameter, split, and metric definition documented for independent reproducibility
- [ ] `LICENSES.md` per §11
- [ ] Working CI smoke test
- [ ] Full results table: all models × all baselines × both metric regimes × both datasets
- [ ] Explicit written discussion of where our reproduction agrees/disagrees with both the original benchmark and the calibration rebuttal, including honest treatment of any discrepancy from published numbers
- [ ] ~1,500–2,500 word technical write-up ready for publication
- [ ] One-page case-study-style PDF summary formatted as a client deliverable would be

---

## References

1. **Ahlmann-Eltze C, Huber W, Anders S.** Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nature Methods.* 2025;22(8):1657–1661. https://doi.org/10.1038/s41592-025-02772-6

2. **Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics.** *bioRxiv.* October 2025. https://www.biorxiv.org/content/10.1101/2025.10.20.683304

3. **Norman TM, Horlbeck MA, Replogle JM, et al.** Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. *Science.* 2019;365:786–793. https://doi.org/10.1126/science.aax4438

4. **Adamson B, et al.** A Multiplexed Single-Cell CRISPR Screening Platform Enables Systematic Dissection of the Unfolded Protein Response. *Cell.* 2016.

5. **Roohani Y, Huang K, Leskovec J.** GEARS: Predicting transcriptional outcomes of novel multi-gene perturbations. *Nature Biotechnology.* 2024. Code: https://github.com/snap-stanford/GEARS

6. **Cui H, Wang C, Maan H, et al.** scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods.* 2024;21:1470–1480. Code: https://github.com/bowang-lab/scGPT

7. **Theodoris CV, et al.** Transfer learning enables predictions in network biology (Geneformer). *Nature.* 2023;618:616–624. Model: https://huggingface.co/ctheodoris/Geneformer

8. **Peidli S, et al.** scPerturb: harmonized single-cell perturbation data. *Nature Methods.* 2024. https://scperturb.org

9. **Csendes G, Sanz G, Szalay KZ, Szalai B.** Benchmarking foundation cell models for post-perturbation RNA-seq prediction. *BMC Genomics.* 2025;26:393. https://doi.org/10.1186/s12864-025-11600-2

10. **Wolf FA, Angerer P, Theis FJ.** SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology.* 2018;19:15.

---

*This BUILD_PLAN.md is a living document. Given this project centers on an active, unresolved scientific debate, expect the reference literature itself to keep evolving during the project's ~9-week build window — check for newer follow-up papers before finalizing the discussion section in Stage 6.*