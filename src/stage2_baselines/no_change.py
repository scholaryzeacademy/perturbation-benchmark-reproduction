"""Stage 2 -- No-change baseline (BUILD_PLAN.md Sec.6 Stage 2, baseline 1).

Predicts that perturbed-cell expression equals control-cell expression --
i.e. zero effect. This is the field's "uninformative" negative-control
baseline. Per BUILD_PLAN.md Sec.9 point 1, sanity-checking this baseline
against published numbers is the first thing to do before trusting anything
built on top of it.
"""
import numpy as np
from anndata import AnnData


class NoChangeBaseline:
    def __init__(self):
        self._ctrl_mean = None

    def fit(self, adata: AnnData, train_conditions: list[str]) -> "NoChangeBaseline":
        # train_conditions is unused -- this baseline only needs control
        # cells -- kept as a parameter so all three baselines in this module
        # share one fit(adata, train_conditions) signature.
        ctrl = adata[adata.obs["condition"] == "ctrl"]
        self._ctrl_mean = np.asarray(ctrl.X.mean(axis=0)).ravel()
        return self

    def predict(self, condition: str) -> np.ndarray:
        if self._ctrl_mean is None:
            raise RuntimeError("call fit() before predict()")
        return self._ctrl_mean
