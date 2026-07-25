"""Downloads scGPT's pretrained 'whole-human' checkpoint (33M human cells,
the checkpoint bowang-lab/scGPT's README recommends for most fine-tuning
uses, including their own perturbation-prediction tutorial). scGPT doesn't
distribute this checkpoint any other way as of this writing -- it's a
Google Drive folder, not a plain URL, so this uses gdown rather than a
simple wget/curl.

Google Drive folder downloads can be flaky (Google sometimes rate-limits or
blocks automated access) -- if this fails, follow the manual fallback it
prints instead of retrying blindly.

Usage:
    python scripts/download_scgpt_checkpoint.py
    python scripts/download_scgpt_checkpoint.py --dest checkpoints/scGPT_human
"""
import argparse
import sys
from pathlib import Path

CHECKPOINT_FOLDER_URL = "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
REQUIRED_FILES = ("args.json", "best_model.pt", "vocab.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="checkpoints/scGPT_human")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    missing = [f for f in REQUIRED_FILES if not (dest / f).exists()]
    if not missing:
        print(f"All expected files already present in {dest}, skipping download.")
        return

    try:
        import gdown
    except ImportError:
        print("gdown isn't installed -- it's part of environment-scgpt.yml's pip deps.")
        sys.exit(1)

    print(f"Downloading scGPT's 'whole-human' checkpoint into {dest} ...")
    print("(This is a ~33M-cell pretrained model; expect this to take a few minutes.)")
    gdown.download_folder(url=CHECKPOINT_FOLDER_URL, output=str(dest), quiet=False, use_cookies=False)

    missing = [f for f in REQUIRED_FILES if not (dest / f).exists()]
    if missing:
        print(
            f"\nAutomatic download did not produce all expected files (missing: {missing}).\n"
            "Google Drive folder downloads are sometimes blocked for automated tools.\n"
            "Manual fallback:\n"
            f"  1. Open {CHECKPOINT_FOLDER_URL} in a browser\n"
            "  2. Download the whole folder (Google Drive: right-click -> Download)\n"
            f"  3. Unzip it so these files end up directly inside {dest}/:\n"
            f"     {', '.join(REQUIRED_FILES)}\n"
        )
        sys.exit(1)

    print(f"Done. Checkpoint ready at {dest}")


if __name__ == "__main__":
    main()
