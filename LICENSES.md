# Licensing & Attribution

Per `docs/BUILD_PLAN.md` §11's licensing checklist. Findings below marked "confirmed" were checked directly against the upstream source (GitHub repo pages / Hugging Face model card) on 2026-07-31, not assumed or web-search-summarized without verification — consistent with this project's general convention of re-deriving facts from source rather than trusting an earlier guess (see `CLAUDE.md`'s Geneformer checkpoint-name correction for a prior example of exactly this kind of mistake being caught and fixed).

## This project's own code

Licensed under the **MIT License** (see `LICENSE` at repo root) — matches both upstream projects this reproduction builds on (GEARS and scGPT, both MIT, below), keeping the whole dependency chain's licensing consistent.

## Upstream code and model weights

| Component | License | Status |
|---|---|---|
| **GEARS** (`snap-stanford/GEARS`, `cell-gears` on PyPI) | MIT | Confirmed directly against the GitHub repo. |
| **scGPT** (`bowang-lab/scGPT`) — code | MIT | Confirmed directly against the GitHub repo. |
| **scGPT** — `whole-human` pretrained checkpoint weights | Not separately stated | **Open item**: the repo's MIT badge covers the code; the pretrained checkpoint (Google-Drive-hosted, fetched via `scripts/download_scgpt_checkpoint.py`) has no license distinct from the code license documented anywhere found during this check. Not assumed to inherit MIT automatically — flag for legal review before any redistribution of the fine-tuned checkpoints this project produced. |
| **Geneformer** (`ctheodoris/Geneformer` on Hugging Face) | **Apache-2.0** | Confirmed directly against the Hugging Face model card. **Not gated** — openly downloadable, no access request or credential needed, as of this check. This corrects an earlier, more cautious note in `CLAUDE.md` ("may be gated/credentialed... confirm before use") that was never actually re-checked against the live page until now. |

## Datasets

| Dataset | Source | Status |
|---|---|---|
| **Norman et al. 2019** (*Science*) Perturb-seq data | Downloaded via GEARS's own `PertData.load(data_name="norman")` | **Open item**: redistribution terms for the dataset itself, as mirrored through GEARS's own download mechanism, were not independently confirmed beyond GEARS's own MIT code license. Recommend checking the original paper's data availability statement (GEO accession) before any redistribution of this project's derived outputs built on it. |
| **Adamson et al. 2016** (*Cell*) Perturb-seq data | Downloaded via GEARS's own `PertData.load(data_name="adamson")` | Same open item as Norman above. |

## Framing papers

This project's entire premise is transparently building on and reproducing both sides of an active, unresolved 2025 debate — not presenting either paper's argument as this project's own original finding. Cited prominently here and in `reports/final_writeup/technical_writeup.md`:

1. Ahlmann-Eltze C, Huber W, Anders S. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nature Methods.* 2025;22(8):1657–1661. https://doi.org/10.1038/s41592-025-02772-6
2. Deep Learning-Based Genetic Perturbation Models Do Outperform Uninformative Baselines on Well-Calibrated Metrics. *bioRxiv.* October 2025. https://www.biorxiv.org/content/10.1101/2025.10.20.683304

Full reference list (including Norman, Adamson, GEARS, scGPT, Geneformer, scPerturb, and metric-benchmarking source papers) is in `docs/BUILD_PLAN.md`'s References section.

## scPerturb

Not used by this project (the optional v2 cross-check extension `docs/BUILD_PLAN.md` mentions was never built) — no terms-of-use confirmation needed.
