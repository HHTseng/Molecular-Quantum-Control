"""Shared chemistry-conditioned GNN action-value function."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from rlqls.multitask.observation import GLOBAL_FEATURE_DIM, MolecularGraphObservation, obs_map

from .chemistry_encoder import ChemistryEncoder
from .pulse_scorer import PulseScorer
from .spectroscopy_encoder import SpectroscopyEncoder


@dataclass(frozen=True, slots=True)
class GNNQConfig:
    hidden_dim: int = 64
    chemistry_context_dim: int = 64
    chemistry_layers: int = 2
    spectroscopy_layers: int = 3
    use_atomistic_chemistry: bool = True
    use_explicit_chemistry: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class ChemistryConditionedQNetwork(nn.Module):
    """Shared ``Q_Theta(o_m,a)`` for heterogeneous molecular tasks.

    The network has no molecule-specific input or output layer.  Graph padding
    fixes tensor shapes, while the same message/scoring parameters are reused
    for all molecules and all valid pulse candidates.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        atom_edge_dim: int,
        explicit_chemistry_dim: int,
        level_dim: int,
        level_edge_dim: int,
        pulse_dim: int,
        pulse_transition_dim: int,
        config: GNNQConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GNNQConfig()
        h = self.config.hidden_dim
        c = self.config.chemistry_context_dim
        self.chemistry_encoder = ChemistryEncoder(
            atom_dim,
            atom_edge_dim,
            explicit_chemistry_dim,
            h,
            c,
            num_layers=self.config.chemistry_layers,
            use_atomistic=self.config.use_atomistic_chemistry,
            use_explicit=self.config.use_explicit_chemistry,
        )
        self.spectroscopy_encoder = SpectroscopyEncoder(
            level_dim,
            level_edge_dim,
            GLOBAL_FEATURE_DIM,
            c,
            h,
            num_layers=self.config.spectroscopy_layers,
        )
        self.pulse_scorer = PulseScorer(
            h,
            pulse_dim,
            pulse_transition_dim,
            h,
            h,
        )

    @staticmethod
    def _ensure_batch(
        observation: MolecularGraphObservation,
    ) -> tuple[MolecularGraphObservation, bool]:
        if observation.atom_features.ndim == 3:
            return observation, False
        if observation.atom_features.ndim != 2:
            raise ValueError("observation must be unbatched or batched")
        return obs_map(lambda leaf: leaf.unsqueeze(0), observation), True

    def forward(
        self,
        observation: MolecularGraphObservation,
        *,
        return_aux: bool = False,
    ):
        obs, squeezed = self._ensure_batch(observation)
        chemistry = self.chemistry_encoder(obs)
        levels, control = self.spectroscopy_encoder(obs, chemistry)
        q_values, pulse_aux = self.pulse_scorer(obs, levels, control)
        if squeezed:
            q_values = q_values.squeeze(0)
            chemistry = chemistry.squeeze(0)
            control = control.squeeze(0)
            levels = levels.squeeze(0)
            pulse_aux = {key: value.squeeze(0) for key, value in pulse_aux.items()}
        if return_aux:
            return q_values, {
                "chemistry_embedding": chemistry,
                "control_embedding": control,
                "level_embeddings": levels,
                **pulse_aux,
            }
        return q_values


__all__ = ["GNNQConfig", "ChemistryConditionedQNetwork"]
