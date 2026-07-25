# Multi-stage Dockerfile — one target per environment in this repo, kept
# separate for the same reasons documented in environment.yml /
# environment-scgpt.yml / environment-geneformer.yml: scGPT's pinned
# `cell-gears<0.0.3` conflicts with the GEARS stage's own `cell-gears`
# (0.1.2), and both GPU stages need a CUDA-matched torch build isolated from
# the CPU-only base (BUILD_PLAN.md §5).
#
# There is no meaningful single default target — always build with --target:
#   docker build --target cpu        -t perturb-bench:cpu .          # Stages 1,2,3a,4,5,6 — also what CI uses
#   docker build --target scgpt      -t perturb-bench:scgpt .        # Stage 3b
#   docker build --target geneformer -t perturb-bench:geneformer .   # Stage 3c
#
# Base images verified to exist on Docker Hub at time of writing:
# mambaorg/micromamba:2-ubuntu22.04, mambaorg/micromamba:2-cuda12.1.1-ubuntu22.04.
# The COPY/RUN pattern below follows micromamba-docker's own documented usage
# (https://micromamba-docker.readthedocs.io/en/latest/advanced_usage.html):
# $MAMBA_USER is the image's non-root default user, and its entrypoint script
# auto-activates the `base` conda env for any CMD run at container start —
# no manual `conda activate` needed.

# ---------------------------------------------------------------------------
# cpu — Stages 1, 2, 3a, 4, 5, 6 (data, baselines, GEARS, conventional
# metrics, calibration, reporting). This is the image the GitHub Actions CI
# smoke test builds and runs against.
# ---------------------------------------------------------------------------
FROM mambaorg/micromamba:2-ubuntu22.04 AS cpu

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install --name base --yes --file /tmp/environment.yml && \
    micromamba clean --all --yes

WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace

EXPOSE 8888
CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# scgpt — Stage 3b (scGPT fine-tuning). GPU-only. pytorch-cuda is pinned to
# 12.1 in environment-scgpt.yml — if the GPU instance actually provisioned in
# BUILD_PLAN.md §10 Phase 0 needs a different CUDA version, update that pin
# and this base image tag together.
# ---------------------------------------------------------------------------
FROM mambaorg/micromamba:2-cuda12.1.1-ubuntu22.04 AS scgpt

COPY --chown=$MAMBA_USER:$MAMBA_USER environment-scgpt.yml /tmp/environment-scgpt.yml
RUN micromamba install --name base --yes --file /tmp/environment-scgpt.yml && \
    micromamba clean --all --yes

WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace

# The optional fast-attention path (flash-attn) is commented out in
# environment-scgpt.yml by design — it compiles against the exact CUDA build
# above and isn't required for correctness, only speed. Uncomment it there
# and rebuild this stage once the GPU/CUDA match is confirmed.

CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# geneformer — Stage 3c (zero-shot in silico perturbation). GPU-only, same
# CUDA-matching note as the scgpt stage above.
# ---------------------------------------------------------------------------
FROM mambaorg/micromamba:2-cuda12.1.1-ubuntu22.04 AS geneformer

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment-geneformer.yml /tmp/environment-geneformer.yml
RUN micromamba install --name base --yes --file /tmp/environment-geneformer.yml && \
    micromamba clean --all --yes

WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace

# Geneformer itself is deliberately NOT cloned/installed at build time: it's
# gated/credentialed on Hugging Face in some versions (BUILD_PLAN.md §11),
# and baking credentials into an image layer is unsafe. After starting a
# container from this image, run:
#   huggingface-cli login   # only if the current repo access requires it
#   git lfs install && git clone https://huggingface.co/ctheodoris/Geneformer /opt/Geneformer
#   pip install /opt/Geneformer

CMD ["/bin/bash"]
