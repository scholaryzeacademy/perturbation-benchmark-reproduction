#!/usr/bin/env bash
set -e

# Trains GEARS (Stage 3a) with this project's real hyperparameters
# (hidden_size=64, epochs=20 -- see configs/training_hyperparameters.yaml)
# on a CUDA GPU. Tested against an RTX 3090 (24GB, Ampere).
#
# HOW TO RUN (open a terminal inside a clone of this repo, then):
#   bash scripts/run_gears_gpu.sh adamson
#   bash scripts/run_gears_gpu.sh norman
# (or just `bash scripts/run_gears_gpu.sh` with no dataset name -- it will
# ask which one)
#
# PREREQUISITES (one-time, before the first run):
#   - conda or miniconda already installed
#   - a working NVIDIA GPU + driver: run `nvidia-smi` in a terminal, it
#     should print your GPU's name. If that command isn't found or errors,
#     stop here and get the NVIDIA driver installed first.
#
# The first run creates a dedicated conda environment (~5-10 minutes, one
# time only). Every run after that reuses it and starts training right away.
# Training itself takes a while (real, non-toy hyperparameters) -- progress
# prints live as it runs, roughly one line per training step.

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

if ! conda env list | grep -q "perturb-bench-gears-gpu"; then
    echo "First run: creating the GPU conda environment (a few minutes)..."
    conda env create -f environment-gears-gpu.yml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate perturb-bench-gears-gpu

echo ""
echo "Starting real GEARS training on '$DATASET'. This downloads the dataset"
echo "on first run, then trains for real (not a quick test) -- expect this"
echo "to take a while, with progress printing as it goes."
echo ""
python -m src.stage3_models.train_gears --dataset "$DATASET"
