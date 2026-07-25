#!/usr/bin/env bash
set -e

# ONE script to set up and run BOTH GPU training jobs (Stage 3a GEARS,
# Stage 3b scGPT) on a fresh remote GPU machine (tested target: RTX 3090).
# Handles everything from an empty machine: installs git/miniconda if
# missing, clones the repo, creates both conda environments, downloads
# both datasets and the scGPT checkpoint, then trains.
#
# HOW TO RUN: SSH into the remote GPU machine, then paste this whole block
# (it backgrounds itself with nohup so closing your laptop / losing the SSH
# connection will NOT kill the run):
#
#   curl -fsSL -o bootstrap_remote_gpu.sh https://raw.githubusercontent.com/scholaryzeacademy/perturbation-benchmark-reproduction/main/scripts/bootstrap_remote_gpu.sh
#   nohup bash bootstrap_remote_gpu.sh adamson > run.log 2>&1 &
#   disown
#   tail -f run.log
#
# Arguments: adamson (default) | norman | both
#   bash bootstrap_remote_gpu.sh both   # runs everything on both datasets
#
# `tail -f run.log` shows live progress; Ctrl+C just stops *watching*, it
# does not stop the run (that's what nohup + disown bought you). To check
# back later from a new SSH session: `cd perturbation-benchmark-reproduction && tail -f run.log`
#
# This runs, in order: GEARS on GPU, then scGPT fine-tuning, for each
# requested dataset. Expect this to take a long time (hours) -- scGPT in
# particular is UNTESTED end-to-end as of this writing (see CLAUDE.md), so
# treat its output as the first real signal of whether it works.

ARG="${1:-adamson}"
if [ "$ARG" = "both" ]; then
    DATASETS="adamson norman"
elif [ "$ARG" = "adamson" ] || [ "$ARG" = "norman" ]; then
    DATASETS="$ARG"
else
    echo "Unrecognized argument '$ARG' -- expected adamson, norman, or both."
    exit 1
fi

REPO_URL="https://github.com/scholaryzeacademy/perturbation-benchmark-reproduction.git"
REPO_DIR="perturbation-benchmark-reproduction"

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

echo "=== Step 1/4: checking for a GPU ==="
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found or failed -- this machine doesn't have a working NVIDIA GPU/driver. Aborting."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Step 2/4: prerequisites (git, conda) ==="
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    $SUDO apt-get update && $SUDO apt-get install -y git
fi

if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda (one-time, a few minutes)..."
    curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    "$HOME/miniconda3/bin/conda" init bash
fi
# shellcheck disable=SC1091
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"

echo "=== Step 3/4: getting the repo ==="
if [ -d "$REPO_DIR" ]; then
    echo "$REPO_DIR already exists, pulling latest instead of re-cloning..."
    (cd "$REPO_DIR" && git pull)
else
    git clone "$REPO_URL"
fi
cd "$REPO_DIR"

echo "=== Step 4/4: training ==="
for DATASET in $DATASETS; do
    echo ""
    echo "----- GEARS (Stage 3a) on $DATASET -----"
    bash scripts/run_gears_gpu.sh "$DATASET"

    echo ""
    echo "----- scGPT (Stage 3b) on $DATASET -----"
    bash scripts/run_scgpt_gpu.sh "$DATASET"
done

echo ""
echo "=== ALL DONE ($DATASETS) ==="
echo "GEARS checkpoints: data/gears_<dataset>_real_ckpt/"
echo "scGPT results:     save/dev_perturb_<dataset>-*/  (best_model.pt, test_metrics.json, run.log)"
