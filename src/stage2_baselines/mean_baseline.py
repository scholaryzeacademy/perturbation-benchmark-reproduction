"""Stage 2 -- Mean baseline (BUILD_PLAN.md Sec.6 Stage 2, baseline 2).

Predicts the mean observed perturbation effect (delta from control) across
the training set, applied uniformly to every held-out test perturbation.
"""
import numpy as np
from anndata import AnnData


class MeanBaseline:
    def __init__(self):
        self._ctrl_mean = None
        self._mean_train_delta = None

    def fit(self, adata: AnnData, train_conditions: list[str]) -> "MeanBaseline":
        ctrl = adata[adata.obs["condition"] == "ctrl"]
        self._ctrl_mean = np.asarray(ctrl.X.mean(axis=0)).ravel()

        deltas = []
        for condition in train_conditions:
            cells = adata[adata.obs["condition"] == condition]
            if cells.n_obs == 0:
                continue
            pert_mean = np.asarray(cells.X.mean(axis=0)).ravel()
            deltas.append(pert_mean - self._ctrl_mean)
        self._mean_train_delta = np.mean(deltas, axis=0)
        return self

    def predict(self, condition: str) -> np.ndarray:
        if self._mean_train_delta is None:
            raise RuntimeError("call fit() before predict()")
        return self._ctrl_mean + self._mean_train_delta
