#!/usr/bin/env bash
set -e

# Fine-tunes scGPT (Stage 3b) for perturbation prediction, following
# scGPT's own documented tutorial, with the real hyperparameters from
# configs/training_hyperparameters.yaml's stage3b_scgpt section. Target
# hardware: Ampere GPUs (RTX 3090 24GB, RTX 3060 12GB). If you hit a CUDA
# out-of-memory error during training on a 12GB-class card, lower
# batch_size/eval_batch_size (default 64/64) in that config section --
# scGPT's own tutorial was tuned assuming more headroom.
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
# The first run creates a dedicated conda environment (a few minutes), then
# compiles flash-attn (~10-30 minutes unless a prebuilt wheel matches your
# exact torch/CUDA/Python combo -- this is normal, not a hang), then
# downloads the pretrained scGPT checkpoint (~a few minutes). Every run
# after that reuses all three and starts training right away.
#
# NOTE: this script is still being shaken out on real hardware -- if
# anything fails, that's genuinely new information, not something to just
# retry past. Report back exactly what printed.

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
    echo "First run: creating the scGPT GPU conda environment (a few minutes)..."
    conda env create -f environment-scgpt.yml
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate perturb-bench-scgpt

# flash-attn is installed here, separately from environment-scgpt.yml, and
# with --no-build-isolation -- its setup.py imports torch at build time to
# detect the CUDA version, and pip's default build isolation hides this
# env's already-installed torch from that build, which fails with
# "ModuleNotFoundError: No module named 'torch'" otherwise (confirmed by a
# real failure). --no-build-isolation makes pip build using this env's
# packages instead of a throwaway one, which is why this step has to run
# after conda activate above, not be listed as a normal dependency.
if ! python -c "import flash_attn" &> /dev/null; then
    echo "Installing flash-attn (compiles from source unless a prebuilt wheel matches your exact torch/CUDA/Python combo -- this can take 10-30+ minutes, that's normal, not a hang)..."
    pip install "flash-attn>=2.8.0" --no-build-isolation
fi

echo ""
echo "Checking for the pretrained scGPT checkpoint..."
python scripts/download_scgpt_checkpoint.py

echo ""
echo "Starting scGPT fine-tuning on '$DATASET'. This downloads the dataset"
echo "on first run, then trains for real (not a quick test) -- expect this"
echo "to take a while, with progress printing as it goes."
echo ""
python -m src.stage3_models.train_scgpt --dataset "$DATASET"
