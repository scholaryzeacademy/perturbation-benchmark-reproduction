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


def _isp_aggregate_grouped_perturb_fixed(cos_sims_df, dict_list, genes_perturbed):
    """Drop-in replacement for
    geneformer.in_silico_perturber_stats.isp_aggregate_grouped_perturb that
    fixes a real bug in the installed library: for a multi-gene group (e.g.
    one of Norman's 2-gene combos), Gene_name is a tuple like
    ('CREB1', 'DAD1'), and upstream's `df["Gene"] = symbol` lets pandas try
    to broadcast that tuple *elementwise* across the per-cell rows, raising
    `ValueError: Length of values (2) does not match length of index (N)`
    for any N != 2. Confirmed via a synthetic 2-gene repro (2026-07-30) --
    Adamson's real test split has no combo conditions to catch this
    natively, but every one of Norman's real combo_seen0/1/2 conditions
    would hit this exact crash in compute_perturbation_stats otherwise.

    Everything here is copied unchanged from the installed
    isp_aggregate_grouped_perturb (verified against the actual installed
    source, not guessed) except the final assignment line, which wraps
    symbol in a list so pandas assigns it as one repeated scalar per row
    instead of trying to unpack it. This is a no-op for the single-gene
    case (where symbol is already a plain string) -- `[symbol] * len(df)`
    and pandas's own scalar broadcast produce an identical column -- so
    this replacement is safe to apply unconditionally, not just for
    multi-gene groups.
    """
    names = ["Cosine_sim", "Gene"]
    cos_sims_full_dfs = []
    if isinstance(genes_perturbed, list):
        if len(genes_perturbed) > 1:
            gene_ids_df = cos_sims_df.loc[
                np.isin(
                    [set(idx) for idx in cos_sims_df["Ensembl_ID"]],
                    set(genes_perturbed),
                ),
                :,
            ]
        else:
            gene_ids_df = cos_sims_df.loc[
                np.isin(cos_sims_df["Ensembl_ID"], genes_perturbed), :
            ]
    else:
        raise ValueError(
            "aggregate_data is for perturbation of single gene or single group of genes. "
            "genes_to_perturb should be formatted as list."
        )

    if gene_ids_df.empty:
        raise ValueError("genes_to_perturb not found in data.")

    tokens = gene_ids_df["Gene"]
    symbols = gene_ids_df["Gene_name"]

    for token, symbol in zip(tokens, symbols):
        cos_shift_data = []
        for dict_i in dict_list:
            cos_shift_data += dict_i.get((token, "cell_emb"), [])

        df = pd.DataFrame(columns=names)
        df["Cosine_sim"] = cos_shift_data
        df["Gene"] = [symbol] * len(df)
        cos_sims_full_dfs.append(df)

    return pd.concat(cos_sims_full_dfs)


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
    falls off the end, so the caller previously got None back).

    Monkeypatches geneformer.in_silico_perturber_stats's own
    isp_aggregate_grouped_perturb with _isp_aggregate_grouped_perturb_fixed
    (mode="aggregate_data" resolves that name as a module-level global at
    call time, so this is enough to redirect it) -- see that function's
    docstring for the real multi-gene bug this works around. Applied
    unconditionally since the fix is a no-op for genes_perturbed of length 1
    (Adamson's case)."""
    import geneformer.in_silico_perturber_stats as isp_stats_module
    from geneformer import InSilicoPerturberStats

    # Verified against ctheodoris/Geneformer commit 04c2b2e84da7c0f385c3f9ad
    # 8f3ec24bab6650e5 (2026-05-26), the HEAD this project's local ./Geneformer
    # clone happened to be on when this patch was written and validated --
    # `pip show geneformer` reports a static "0.1.0" that doesn't track this,
    # so the commit hash is the real version signal here. NOTE:
    # environment-geneformer.yml does NOT pin a commit (unlike
    # environment-scgpt.yml, which does) -- a fresh `git clone` following its
    # instructions can land on a newer commit than this one, silently.
    # If that ever happens, re-check upstream's
    # in_silico_perturber_stats.py:isp_aggregate_grouped_perturb for whether
    # `df["Gene"] = symbol` has been fixed there directly -- if so, this
    # monkeypatch (and _isp_aggregate_grouped_perturb_fixed above) can likely
    # be deleted rather than silently going stale.
    isp_stats_module.isp_aggregate_grouped_perturb = _isp_aggregate_grouped_perturb_fixed

    Path(output_directory).mkdir(parents=True, exist_ok=True)
    stats = InSilicoPerturberStats(
        mode=mode,
        genes_perturbed=genes_perturbed,
        model_version=model_version,
    )
    stats.get_stats(input_directory, None, output_directory, output_prefix)
    output_path = (Path(output_directory) / output_prefix).with_suffix(".csv")
    return pd.read_csv(output_path)
