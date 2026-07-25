"""Stage 3a -- GEARS (specialized graph-based model) (BUILD_PLAN.md Sec.6 Stage 3a).

Trains GEARS using its own documented training procedure and default
hyperparameters -- GEARS is meant to represent the standard, well-packaged
DL specialist baseline in this comparison, not a hand-tuned model. The
"real" hyperparameter values (used for actual reported results) live in
configs/training_hyperparameters.yaml's stage3a_gears section and are
GEARS's own library defaults, not ours.

Unlike the Stage 2 baselines, fit() takes the full PertData object rather
than a plain (adata, train_conditions) pair: GEARS's model genuinely needs
the graph-structured dataloaders PertData.get_dataloader() builds (gene
co-expression / GO similarity graphs), which a bare AnnData can't provide.
That is a real difference in what the model needs, not an inconsistency to
paper over -- see CLAUDE.md's "working conventions" section.
"""
import numpy as np
import torch
from gears import GEARS, PertData

from src.perturbation_conditions import perturbed_genes


class GearsModel:
    def __init__(
        self,
        device: str | None = None,
        hidden_size: int = 64,
        epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        weight_bias_track: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.weight_bias_track = weight_bias_track
        self._model = None

    def fit(self, pert_data: PertData) -> "GearsModel":
        self._model = GEARS(pert_data, device=self.device, weight_bias_track=self.weight_bias_track)
        self._model.model_initialize(hidden_size=self.hidden_size)
        self._model.train(epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay)
        return self

    def load(self, pert_data: PertData, path: str) -> "GearsModel":
        """Load a checkpoint saved by save() instead of training from scratch --
        useful given BUILD_PLAN.md Sec.8 flags this as the most compute- and
        debugging-intensive phase; avoids re-training just to re-evaluate."""
        self._model = GEARS(pert_data, device=self.device, weight_bias_track=self.weight_bias_track)
        self._model.load_pretrained(path)
        return self

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("call fit() or load() before save()")
        self._model.save_model(path)

    def predict_many(self, conditions: list[str]) -> dict[str, np.ndarray]:
        """Batched prediction -- GEARS's own predict() takes a list of
        perturbations and builds one DataLoader for all of them, far cheaper
        than calling predict() once per condition for a large test set."""
        if self._model is None:
            raise RuntimeError("call fit() or load() before predict")
        pert_list = [perturbed_genes(c) for c in conditions]
        raw = self._model.predict(pert_list)
        return {condition: raw["_".join(genes)] for condition, genes in zip(conditions, pert_list)}

    def predict(self, condition: str) -> np.ndarray:
        return self.predict_many([condition])[condition]
