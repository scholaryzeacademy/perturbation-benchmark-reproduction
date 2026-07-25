#!/usr/bin/env bash
set -e

# Fine-tunes scGPT (Stage 3b) for perturbation prediction, following
# scGPT's own documented tutorial, with the real hyperparameters from
# configs/training_hyperparameters.yaml's stage3b_scgpt section. Tested
# against an RTX 3090 (24GB, Ampere).
#
# HOW TO RUN (open a terminal inside a clone of this repo, then):
#   bash scripts/run_scgpt_gpu.sh adamson
#   bash scripts/run_scgpt_gpu.sh norman
# (or just `bash scripts/run_scgpt_gpu.sh` with no dataset name -- it will
# ask which one)
#
# PREREQUISITES (one-time, before the first run):
#   - conda or miniconda already installed
#   - a working NVIDIA GPU + driver: run `nvidia-smi` in a terminal, it
#     should print your GPU's name. If that command isn't found or errors,
#     stop here and get the NVIDIA driver installed first.
#   - at least ~10GB free disk space (pretrained checkpoint + dataset)
#
# The first run creates a dedicated conda environment (~10-30 minutes --
# flash-attn compiles from source unless a prebuilt wheel matches your exact
# torch/CUDA/Python combo, this is normal, not a hang) and downloads the
# pretrained scGPT checkpoint (~a few minutes). Every run after that reuses
# both and starts training right away.
#
# NOTE: unlike scripts/run_gears_gpu.sh, this script has not been run
# end-to-end on real hardware yet -- if anything fails on your first run,
# that's genuinely new information, not something to just retry past.

DATASET="$1"
if [ -z "$DATASET" ]; then
    echo "Which dataset do you want to train on? Type adamson or norman, then press Enter:"
    read -r DATASET
fi

if [ "$DATASET" != "adamson" ] && [ "$DATASET" != "norman" ]; then
    echo "Unrecognized dataset '$DATASET' -- expected 'adamson' or 'norman'."
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda &> /dev/null; then
    echo "conda was not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "nvidia-smi failed -- no working NVIDIA GPU/driver detected. This script needs a GPU machine."
    exit 1
fi

if ! conda env list | grep -q "perturb-bench-scgpt"; then
    echo "First run: creating the scGPT GPU conda environment (10-30 minutes, mostly flash-attn compiling)..."
    conda env create -f environment-scgpt.yml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate perturb-bench-scgpt

echo ""
echo "Checking for the pretrained scGPT checkpoint..."
python scripts/download_scgpt_checkpoint.py

echo ""
echo "Starting scGPT fine-tuning on '$DATASET'. This downloads the dataset"
echo "on first run, then trains for real (not a quick test) -- expect this"
echo "to take a while, with progress printing as it goes."
echo ""
python -m src.stage3_models.train_scgpt --dataset "$DATASET"
