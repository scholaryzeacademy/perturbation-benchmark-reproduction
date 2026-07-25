"""Stage 3b -- scGPT fine-tuning for perturbation prediction (BUILD_PLAN.md
Sec.6 Stage 3b): "follow scGPT's documented perturbation-prediction
fine-tuning example, fine-tuning on the same training split as GEARS for
direct comparability."

This is a close, minimally-adapted port of bowang-lab/scGPT's own
tutorials/Tutorial_Perturbation.ipynb (verified directly against that
notebook, not written from memory) -- CLI args + configs/training_hyperparameters.yaml's
stage3b_scgpt section replace the notebook's hardcoded settings cells, but
the actual data loading / model / training / evaluation logic is kept as
close to their source as possible, since this is scGPT's own documented
procedure, not something to redesign.

Needs the perturb-bench-scgpt conda env (environment-scgpt.yml) and the
pretrained scGPT checkpoint (see scripts/download_scgpt_checkpoint.py) --
NOT the base perturb-bench env, which deliberately excludes scGPT (see
CLAUDE.md's Environment setup section for why).

Unlike train_gears.py, this has NOT been execution-tested end-to-end (no
GPU with enough VRAM/flash-attn support was available while writing it) --
treat the first real run as the actual test of this script, and report
back anything that breaks.

Usage:
    python -m src.stage3_models.train_scgpt --dataset adamson
    python -m src.stage3_models.train_scgpt --dataset norman
"""
import argparse
import copy
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe: this runs on remote GPU machines with no display

import numpy as np
import torch
import yaml
from torch import nn
from torch_geometric.loader import DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

from gears import PertData
from gears.inference import deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import masked_mse_loss
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id, compute_perturbation_metrics

from src.stage1_data.load_data import CONFIG_PATH, SUPPORTED_DATASETS

PERTS_TO_PLOT = {"adamson": "KCTD16+ctrl", "norman": "SAMD1+ZBTB1"}


def train_one_epoch(model, train_loader, optimizer, scaler, scheduler, criterion, cfg, gene_ids, n_genes, device, logger, epoch):
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()
    num_batches = len(train_loader)

    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y

        input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        if len(input_gene_ids) > cfg["max_seq_len"]:
            input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[: cfg["max_seq_len"]]
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
        src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)

        with torch.cuda.amp.autocast(enabled=cfg["amp"]):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            output_values = output_dict["mlm_output"]
            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=False if scaler.is_enabled() else True
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient scaler. "
                    f"The current scale is {scaler.get_scale()}. This warning can be ignored "
                    "if it no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % 100 == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / 100
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {total_loss / 100:5.2f} | mse {total_mse / 100:5.2f} |"
            )
            total_loss, total_mse = 0.0, 0.0
            start_time = time.time()


def eval_perturb(loader, model, include_zero_gene, gene_ids, device) -> Dict:
    model.eval()
    model.to(device)
    pert_cat, pred, truth, pred_de, truth_de = [], [], [], [], []

    for batch in loader:
        batch.to(device)
        pert_cat.extend(batch.pert)
        with torch.no_grad():
            p = model.pred_perturb(batch, include_zero_gene=include_zero_gene, gene_ids=gene_ids)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            for i, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[i, de_idx])
                truth_de.append(t[i, de_idx])

    results = {"pert_cat": np.array(pert_cat)}
    pred, truth = torch.stack(pred), torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(float)
    results["truth"] = truth.detach().cpu().numpy().astype(float)
    pred_de, truth_de = torch.stack(pred_de), torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(float)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)
    return results


def predict(model, pert_list, pert_data, gene_ids, include_zero_gene, eval_batch_size, amp, device, pool_size: Optional[int] = None) -> Dict:
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    for pert in pert_list:
        for g in pert:
            if g not in gene_list:
                raise ValueError(f"{g} is not in the perturbation graph. Select from GEARS.gene_list!")

    model.eval()
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(pert, ctrl_adata, gene_list, device, num_samples=pool_size)
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp)
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)
    return results_pred


def plot_perturbation(model, query, pert_data, gene_ids, cfg, device, save_file: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)
    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [gene2idx[gene_raw2id[i]] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]]
    genes = [gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]]
    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]

    if query.split("+")[1] == "ctrl":
        pred = predict(model, [[query.split("+")[0]]], pert_data, gene_ids, cfg["include_zero_gene"], cfg["eval_batch_size"], cfg["amp"], device, pool_size=300)
        pred = pred[query.split("+")[0]][de_idx]
    else:
        pred = predict(model, [query.split("+")], pert_data, gene_ids, cfg["include_zero_gene"], cfg["eval_batch_size"], cfg["amp"], device, pool_size=300)
        pred = pred["_".join(query.split("+"))][de_idx]

    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values
    pred, truth = pred - ctrl_means, truth - ctrl_means

    fig, ax = plt.subplots(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))
    for i in range(pred.shape[0]):
        plt.scatter(i + 1, pred[i], color="red")
    plt.axhline(0, linestyle="dashed", color="green")
    ax.xaxis.set_ticklabels(genes, rotation=90)
    plt.ylabel("Change in Gene Expression over Control", labelpad=10)
    sns.despine()
    fig.savefig(save_file, bbox_inches="tight", transparent=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--data-dir", default="./data_scgpt", help="Separate from the base env's ./data -- scGPT's env pins an older cell-gears (<0.0.3) whose on-disk cache format isn't verified compatible with the 0.1.2 cache the base env writes.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/scGPT_human")
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detects if omitted")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["stage3b_scgpt"]

    set_seed(cfg["seed"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    save_dir = Path(f"./save/dev_perturb_{args.dataset}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {save_dir}", flush=True)

    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}, device={device}")

    print(f"=== Loading {args.dataset!r} (full dataset, real split config) ===", flush=True)
    pert_data = PertData(args.data_dir)
    pert_data.load(data_name=args.dataset)
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=cfg["batch_size"], test_batch_size=cfg["eval_batch_size"])

    model_dir = Path(args.checkpoint_dir)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"
    for f in (model_config_file, model_file, vocab_file):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found -- run scripts/download_scgpt_checkpoint.py first.")

    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [1 if g in vocab else -1 for g in pert_data.adata.var["gene_name"]]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}.")
    genes = pert_data.adata.var["gene_name"].tolist()

    with open(model_config_file) as f:
        model_configs = json.load(f)
    logger.info(f"Resuming model from {model_file}; model args override configs/training_hyperparameters.yaml's stage3b_scgpt architecture fields.")
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=int)
    n_genes = len(genes)

    print(f"=== Building TransformerGenerator (embsize={embsize}, nlayers={nlayers}) ===", flush=True)
    model = TransformerGenerator(
        len(vocab), embsize, nhead, d_hid, nlayers,
        nlayers_cls=n_layers_cls, n_cls=1, vocab=vocab,
        dropout=cfg["dropout"], pad_token="<pad>", pad_value=0,
        pert_pad_id=0, use_fast_transformer=cfg["use_fast_transformer"],
    )
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if any(k.startswith(p) for p in cfg["load_param_prefixs"])}
    for k, v in pretrained_dict.items():
        logger.info(f"Loading pretrained param {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg["schedule_interval"], gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"])

    print(f"=== Training for up to {cfg['epochs']} epochs (early_stop={cfg['early_stop']}) ===", flush=True)
    best_val_corr, best_model, patience = 0.0, None, 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()
        train_one_epoch(
            model, pert_data.dataloader["train_loader"], optimizer, scaler, scheduler,
            criterion, cfg, gene_ids, n_genes, device, logger, epoch,
        )
        val_res = eval_perturb(pert_data.dataloader["val_loader"], model, cfg["include_zero_gene"], gene_ids, device)
        val_metrics = compute_perturbation_metrics(val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"])
        logger.info(f"val_metrics at epoch {epoch}: {val_metrics}")
        logger.info(f"| end of epoch {epoch:3d} | time: {time.time() - epoch_start:5.2f}s |")

        if val_metrics["pearson"] > best_val_corr:
            best_val_corr = val_metrics["pearson"]
            best_model = copy.deepcopy(model)
            logger.info(f"Best model so far, val pearson={best_val_corr:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= cfg["early_stop"]:
                logger.info(f"Early stop at epoch {epoch}")
                break
        scheduler.step()

    print(f"=== Training done in {time.time() - t0:.1f}s ===", flush=True)
    if best_model is None:
        best_model = model
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    print(f"=== Saved checkpoint to {save_dir / 'best_model.pt'} ===", flush=True)

    plot_query = PERTS_TO_PLOT.get(args.dataset)
    if plot_query is not None:
        try:
            plot_perturbation(best_model, plot_query, pert_data, gene_ids, cfg, device, str(save_dir / f"{plot_query}.png"))
        except Exception as e:  # cosmetic only -- don't let a plotting issue hide real results below
            logger.warning(f"Skipping example plot for {plot_query!r}: {e}")

    print("=== Evaluating on held-out test conditions ===", flush=True)
    test_res = eval_perturb(pert_data.dataloader["test_loader"], best_model, cfg["include_zero_gene"], gene_ids, device)
    test_metrics = compute_perturbation_metrics(test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"])
    print(test_metrics, flush=True)
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f)

    deeper_res = deeper_analysis(pert_data.adata, test_res)
    non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)
    metrics = ["pearson_delta", "pearson_delta_de"]
    metrics_non_dropout = ["pearson_delta_top20_de_non_dropout", "pearson_top20_de_non_dropout"]

    subgroup_analysis: Dict[str, Dict[str, List[float]]] = {
        name: {m: [] for m in metrics + metrics_non_dropout} for name in pert_data.subgroup["test_subgroup"].keys()
    }
    for name, pert_list in pert_data.subgroup["test_subgroup"].items():
        for pert in pert_list:
            for m in metrics:
                subgroup_analysis[name][m].append(deeper_res[pert][m])
            for m in metrics_non_dropout:
                subgroup_analysis[name][m].append(non_dropout_res[pert][m])

    for name, result in subgroup_analysis.items():
        for m, values in result.items():
            logger.info(f"test_{name}_{m}: {np.mean(values)}")

    print(f"=== DONE. Results in {save_dir} ===", flush=True)


if __name__ == "__main__":
    main()
