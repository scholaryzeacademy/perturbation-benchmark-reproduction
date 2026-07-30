"""Stage 3c -- Geneformer zero-shot in silico perturbation (BUILD_PLAN.md Sec.6
Stage 3c).

Zero-shot only: no fine-tuning, no ridge decoder on top of Geneformer's
embeddings. Verified directly against ctheodoris/Geneformer's own
geneformer/tokenizer.py, geneformer/in_silico_perturber.py, and
geneformer/in_silico_perturber_stats.py source (not guessed) -- see
BUILD_PLAN.md Sec.6 Stage 3c's honest-scope note for why this project uses
Geneformer's public InSilicoPerturber/InSilicoPerturberStats cosine-shift API
rather than porting Ahlmann-Eltze et al.'s embedding+ridge-decoder approach
(which needs undocumented internal token-manipulation/embedding-extraction
functions with no official example to verify a port against, unlike GEARS's
and scGPT's ports).

Output shape is therefore fundamentally different from GEARS/scGPT: a
per-gene cosine-shift ranking (how much perturbing this gene moves cell
embeddings), not a predicted post-perturbation expression profile. There is
deliberately no fit()/predict(condition)->np.ndarray here matching Stage 2's
contract -- there is nothing to fit (zero-shot) and no expression vector to
return, the same reasoning CLAUDE.md documents for why train_scgpt.py is a
script rather than a Stage-2-shaped class, one level further.

genes_to_perturb passed as a *list* of Ensembl IDs perturbs every gene in
that list *together as one combined group perturbation in every cell* --
confirmed directly from in_silico_perturber.py's __init__ (a list sets
perturb_group=True and forces combos=0/anchor_gene=None, overriding whatever
combos was passed). That is actually the right primitive for this project:
call run_in_silico_perturbation() once per test condition, passing a
single-element list for Adamson's single-gene knockdowns or a two-element
list for Norman's double-gene combos -- each call becomes exactly that one
condition's perturbation, individually or jointly as appropriate. Do NOT
pass a list of every test gene in one call expecting per-gene results back;
that perturbs them all at once instead.

KNOWN LIMITATION -- no true raw per-cell counts available: Geneformer's
TranscriptomeTokenizer expects adata.obs['n_counts'] to hold each cell's
real, varying total UMI count, used to depth-normalize expression before
rank-encoding genes. GEARS's PertData.h5ad (this project's Stage 1 data
source for every other stage) only ships post-normalize_total + log1p
expression with no raw-counts layer -- confirmed directly against
data/adamson/perturb_processed.h5ad and data/norman/perturb_processed.h5ad
(both have `.raw is None` and empty `.layers`). Reconstructing counts via
expm1(X) gives values that sum to a near-constant ~5000-5200 per cell (std
~3% of the mean, checked on Adamson) across all cells -- i.e. the *original*
varying sequencing depth was already normalized away upstream and is not
recoverable from this file. Using that near-constant reconstruction as
n_counts is a documented approximation, not real data: total-count
normalization is a per-cell scalar multiply, so it preserves each cell's
*within-cell* gene rank order (the dominant signal Geneformer's rank-value
encoding uses) but does not reproduce the scale Geneformer's own
gene_median_file was calibrated against from its real training corpus. If
Stage 3c's results look implausible on the first real run, check this before
assuming the model or code itself is at fault -- it's a real, checked
limitation of the input data available to this project, not a hidden
shortcut.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData


def add_geneformer_columns(adata: AnnData) -> AnnData:
    """Add the two columns TranscriptomeTokenizer requires that GEARS's
    PertData.h5ad doesn't ship: adata.var['ensembl_id'] and
    adata.obs['n_counts']. Mutates and returns adata.

    adata.var.index is already Ensembl gene IDs in GEARS's processed h5ad
    (verified: e.g. 'ENSG00000228463') -- no external symbol->Ensembl mapping
    needed. n_counts is reconstructed via expm1(X).sum(axis=1); see this
    module's docstring for why that's a documented approximation, not real
    per-cell sequencing depth.
    """
    adata.var["ensembl_id"] = adata.var.index
    X = adata.X
    dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    adata.obs["n_counts"] = np.expm1(dense).sum(axis=1)
    return adata


def symbols_to_ensembl(adata: AnnData, symbols: list[str]) -> list[str]:
    """Map gene symbols (as used in adata.obs['condition']/GEARS condition
    strings, e.g. 'IER3IP1') to their Ensembl IDs via adata.var['gene_name'],
    which GEARS's processed h5ad already carries alongside the Ensembl-ID
    var index. Raises KeyError naming the missing symbol(s) rather than
    silently dropping them."""
    name_to_id = {name: idx for idx, name in zip(adata.var.index, adata.var["gene_name"])}
    missing = [s for s in symbols if s not in name_to_id]
    if missing:
        raise KeyError(f"gene symbol(s) not found in adata.var['gene_name']: {missing}")
    return [name_to_id[s] for s in symbols]


def tokenize_control_cells(
    adata: AnnData,
    output_dir: str,
    output_prefix: str,
    model_input_size: int = 4096,
    model_version: str = "V2",
    nproc: int = 4,
) -> str:
    """Tokenize the control ('unperturbed') cells -- these are the substrate
    in silico perturbation is applied to, per Geneformer's own documented
    usage pattern (simulate a perturbation on real control cells, since
    zero-shot means there's no fine-tuning step that could otherwise learn
    directly from real perturbed ones). Filters on obs['condition'] == 'ctrl',
    the same convention src/perturbation_conditions.py and GEARS's own
    PertData.ctrl_adata use. Returns the path to the resulting .dataset
    directory.
    """
    from geneformer import TranscriptomeTokenizer

    ctrl = adata[adata.obs["condition"] == "ctrl"].copy()
    add_geneformer_columns(ctrl)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="geneformer_ctrl_"))
    try:
        ctrl.write_h5ad(tmp_dir / f"{output_prefix}.h5ad")
        tokenizer = TranscriptomeTokenizer(
            nproc=nproc,
            model_input_size=model_input_size,
            model_version=model_version,
        )
        tokenizer.tokenize_data(
            data_directory=tmp_dir,
            output_directory=output_dir,
            output_prefix=output_prefix,
            file_format="h5ad",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return str(Path(output_dir) / f"{output_prefix}.dataset")


def run_in_silico_perturbation(
    model_directory: str,
    tokenized_data_path: str,
    genes_to_perturb: list[str],
    output_directory: str,
    output_prefix: str,
    perturb_type: str = "delete",
    emb_mode: str = "cls",
    cell_emb_style: str = "mean_pool",
    forward_batch_size: int = 16,
    nproc: int = 4,
    model_version: str = "V2",
    max_ncells: int | None = None,
) -> None:
    """Run Geneformer's own in silico perturbation (delete/overexpress) on
    the tokenized cells, writing cosine-shift-impact pickles to
    output_directory. genes_to_perturb is perturbed as ONE combined group in
    every cell (see module docstring) -- pass a single-element list for a
    single-gene test condition, or a two-element list for one of Norman's
    double-gene combos. Call this once per test condition, not once for the
    whole gene set.

    emb_mode defaults to "cls", not "cell": V2 checkpoints tokenize with a
    leading <cls> token, and InSilicoPerturber.perturb_data hard-requires
    emb_mode to contain "cls" whenever the first token is <cls>, raising
    otherwise (confirmed by a real crash with emb_mode="cell" against a V2
    checkpoint). Only use "cell"/"cell_and_gene" against a V1 checkpoint
    (no <cls> token).

    Creates output_directory if it doesn't exist -- InSilicoPerturber.perturb_data
    doesn't do this itself and raises FileNotFoundError partway through a run
    otherwise (confirmed by a real crash: unlike tokenize_control_cells's
    output_dir, nothing upstream of this call was creating isp_raw/).
    """
    from geneformer import InSilicoPerturber

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    isp = InSilicoPerturber(
        perturb_type=perturb_type,
        genes_to_perturb=genes_to_perturb,
        model_type="Pretrained",
        emb_mode=emb_mode,
        cell_emb_style=cell_emb_style,
        max_ncells=max_ncells,
        forward_batch_size=forward_batch_size,
        nproc=nproc,
        model_version=model_version,
    )
    isp.perturb_data(model_directory, tokenized_data_path, output_directory, output_prefix)


def compute_perturbation_stats(
    input_directory: str,
    output_directory: str,
    output_prefix: str,
    genes_perturbed: list[str],
    mode: str = "aggregate_data",
    model_version: str = "V2",
):
    """Aggregate the per-cell cosine-shift pickles from
    run_in_silico_perturbation (for the SAME genes_perturbed group used in
    that call) into a per-gene/per-group cosine-shift summary CSV written to
    output_directory. Creates output_directory if it doesn't exist, same
    reasoning as run_in_silico_perturbation above.

    Returns that CSV re-read as a pandas DataFrame -- InSilicoPerturberStats
    .get_stats() itself has no return statement in the installed version
    (confirmed by a real crash: it only calls cos_sims_df.to_csv(...) and
    falls off the end, so the caller previously got None back)."""
    from geneformer import InSilicoPerturberStats

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    stats = InSilicoPerturberStats(
        mode=mode,
        genes_perturbed=genes_perturbed,
        model_version=model_version,
    )
    stats.get_stats(input_directory, None, output_directory, output_prefix)
    output_path = (Path(output_directory) / output_prefix).with_suffix(".csv")
    return pd.read_csv(output_path)
