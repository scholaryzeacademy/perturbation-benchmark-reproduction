"""Stage 4 -- evaluate the fine-tuned scGPT checkpoint on real held-out test
conditions, via the same metrics.evaluate_model() contract
run_baselines_gears.py uses (BUILD_PLAN.md Sec.6 Stage 4).

Runs in the perturb-bench-scgpt env -- NOT perturb-bench-gears-gpu, since
scGPT needs an older cell-gears pin incompatible with GEARS's own (see
CLAUDE.md's Environment setup section). metrics.py itself has no
gears/scgpt import, so this driver and run_baselines_gears.py share the
exact same MSE@20DEG/Pearson(delta control) code from different envs.

Loads the fine-tuned checkpoint train_scgpt.py already saved
(save/dev_perturb_{dataset}-*/best_model.pt) -- no retraining. Reconstructs
the same TransformerGenerator architecture train_scgpt.py builds (using the
PRETRAINED checkpoint's own args.json/vocab.json for architecture/vocab,
same as training did), then loads the fine-tuned state_dict directly
(strict=True): unlike loading the ORIGINAL pretrained checkpoint (which
needs scgpt.utils.load_pretrained's partial/renamed-key matching for the
flash-attn-vs-fallback naming mismatch, see CLAUDE.md), the fine-tuned
checkpoint was saved via `torch.save(best_model.state_dict(), ...)` from
this exact already-built architecture, so every key should already match.

Reuses train_scgpt.py's own module-level predict() function unchanged --
wrapped into predict(condition) -> np.ndarray for evaluate_model().

--max-conditions/--pool-size let you smoke-test on a small subset before
committing to a full run, same reasoning as run_geneformer.py's own such
flags.

Usage:
    python -m src.stage4_conventional_metrics.run_scgpt --dataset adamson
    python -m src.stage4_conventional_metrics.run_scgpt --dataset norman
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from gears import PertData

from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab

from src.perturbation_conditions import perturbed_genes
from src.stage1_data.load_data import CONFIG_PATH, SUPPORTED_DATASETS
from src.stage3_models.train_scgpt import predict as scgpt_predict
from src.stage4_conventional_metrics.metrics import evaluate_model


def _find_finetuned_checkpoint(dataset: str) -> Path:
    """train_scgpt.py timestamps each run's save dir
    (save/dev_perturb_{dataset}-{time}/), so there's no fixed path -- pick
    the most recently modified matching dir that actually contains a
    best_model.pt. Matching by "has best_model.pt", not just name prefix,
    skips earlier incomplete/aborted attempts for the same dataset (e.g.
    save/dev_perturb_adamson-Jul30-03-43/ has only a run.log, no checkpoint
    -- from before the real flash-attn/checkpoint-loading bug fix CLAUDE.md
    documents; the real run is the later save/dev_perturb_adamson-Jul30-03-51/).
    """
    candidates = [
        p for p in Path("save").glob(f"dev_perturb_{dataset}-*") if (p / "best_model.pt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No save/dev_perturb_{dataset}-*/best_model.pt found -- run train_scgpt.py --dataset {dataset} first."
        )
    best_dir = max(candidates, key=lambda p: (p / "best_model.pt").stat().st_mtime)
    return best_dir / "best_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-dir", default="./data_scgpt", help="separate from the base ./data -- same reasoning as train_scgpt.py's own --data-dir default")
    parser.add_argument("--checkpoint-dir", default="checkpoints/scGPT_human", help="the PRETRAINED checkpoint dir (architecture config/vocab source), not the fine-tuned one (found automatically)")
    parser.add_argument("--finetuned-checkpoint", default=None, help="defaults to the most recent save/dev_perturb_{dataset}-*/best_model.pt that actually has a checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="./data/stage4_results")
    parser.add_argument("--max-conditions", type=int, default=None, help="only evaluate the first N test conditions (smoke-testing)")
    parser.add_argument("--pool-size", type=int, default=None, help="override the number of control cells used as the prediction substrate per condition (smoke-testing); default is scgpt_predict's own default, all control cells")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["stage3b_scgpt"]

    finetuned_path = Path(args.finetuned_checkpoint) if args.finetuned_checkpoint else _find_finetuned_checkpoint(args.dataset)
    print(f"=== Using fine-tuned checkpoint: {finetuned_path} ===", flush=True)

    print(f"=== Loading {args.dataset!r} (full dataset, real split config) ===", flush=True)
    t0 = time.time()
    pert_data = PertData(args.data_dir)
    pert_data.load(data_name=args.dataset)
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=cfg["batch_size"], test_batch_size=cfg["eval_batch_size"])
    print(f"=== Data ready in {time.time() - t0:.1f}s ===", flush=True)

    model_dir = Path(args.checkpoint_dir)
    model_config_file = model_dir / "args.json"
    vocab_file = model_dir / "vocab.json"

    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    genes = pert_data.adata.var["gene_name"].tolist()
    with open(model_config_file) as f:
        model_configs = json.load(f)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=int)

    print(f"=== Building TransformerGenerator (embsize={embsize}, nlayers={nlayers}) ===", flush=True)
    model = TransformerGenerator(
        len(vocab), embsize, nhead, d_hid, nlayers,
        nlayers_cls=n_layers_cls, n_cls=1, vocab=vocab,
        dropout=cfg["dropout"], pad_token="<pad>", pad_value=0,
        pert_pad_id=0, use_fast_transformer=cfg["use_fast_transformer"],
    )
    state_dict = torch.load(finetuned_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"=== Loaded fine-tuned weights from {finetuned_path} ===", flush=True)

    def predict_fn(condition: str) -> np.ndarray:
        genes_list = perturbed_genes(condition)
        result = scgpt_predict(
            model, [genes_list], pert_data, gene_ids,
            cfg["include_zero_gene"], cfg["eval_batch_size"], cfg["amp"], device,
            pool_size=args.pool_size,
        )
        return result["_".join(genes_list)]

    adata = pert_data.adata
    test_conditions = [c for c in pert_data.set2conditions["test"] if c != "ctrl"]
    if args.max_conditions is not None:
        test_conditions = test_conditions[: args.max_conditions]
    print(f"=== {len(test_conditions)} test conditions ===", flush=True)

    t0 = time.time()
    df = evaluate_model(predict_fn, adata, test_conditions, model_name="scgpt")
    df["dataset"] = args.dataset
    print(
        f"[scgpt] mean MSE@20DEG={df['mse_20deg'].mean():.4f} "
        f"mean Pearson(dctrl)={df['pearson_delta_control'].mean():.4f} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    out_path = output_dir / f"{args.dataset}_scgpt.csv"
    df.to_csv(out_path, index=False)
    print(f"=== Saved {out_path} ===", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
