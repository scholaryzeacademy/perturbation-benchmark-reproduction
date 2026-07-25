"""Stage 2 -- Ridge regression baseline (BUILD_PLAN.md Sec.6 Stage 2, baseline 3).

Fits scikit-learn Ridge regression to predict each condition's expression
delta from control, using a multi-hot encoding of the perturbed gene(s) as
the "perturbation embedding" input -- BUILD_PLAN.md Sec.5 describes this
baseline as ridge regression "from perturbed embeddings", following
Ahlmann-Eltze et al.'s methodology. A multi-hot gene encoding is a
deliberately simple stand-in for a learned/pretrained embedding (e.g. ESM-2
protein embeddings); this is documented here explicitly as a scope choice,
not a hidden shortcut.

A direct consequence of this simple encoding: a test condition whose gene(s)
never appeared in any training condition encodes as an all-zero vector, so
the model falls back to (approximately) its intercept term -- a linear model
over gene identity cannot extrapolate to a never-seen gene without a richer
embedding. That is expected, not a bug.
"""
import numpy as np
from anndata import AnnData
from sklearn.linear_model import Ridge


def _perturbed_genes(condition: str) -> list[str]:
    return [g for g in condition.split("+") if g != "ctrl"]


class RidgeBaseline:
    def __init__(self, alpha: float = 1.0, seed: int = 1):
        self.alpha = alpha
        self.seed = seed
        self._gene_vocab = None
        self._ctrl_mean = None
        self._model = None

    def _encode(self, condition: str) -> np.ndarray:
        vec = np.zeros(len(self._gene_vocab), dtype=float)
        for gene in _perturbed_genes(condition):
            if gene in self._gene_vocab:
                vec[self._gene_vocab[gene]] = 1.0
        return vec

    def fit(self, adata: AnnData, train_conditions: list[str]) -> "RidgeBaseline":
        ctrl = adata[adata.obs["condition"] == "ctrl"]
        self._ctrl_mean = np.asarray(ctrl.X.mean(axis=0)).ravel()

        genes = sorted({g for c in train_conditions for g in _perturbed_genes(c)})
        self._gene_vocab = {gene: i for i, gene in enumerate(genes)}

        X, y = [], []
        for condition in train_conditions:
            cells = adata[adata.obs["condition"] == condition]
            if cells.n_obs == 0:
                continue
            pert_mean = np.asarray(cells.X.mean(axis=0)).ravel()
            X.append(self._encode(condition))
            y.append(pert_mean - self._ctrl_mean)

        self._model = Ridge(alpha=self.alpha, random_state=self.seed)
        self._model.fit(np.vstack(X), np.vstack(y))
        return self

    def predict(self, condition: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")
        delta = self._model.predict(self._encode(condition).reshape(1, -1))[0]
        return self._ctrl_mean + delta
