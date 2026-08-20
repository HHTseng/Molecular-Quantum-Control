"""Molecule-specific task records sharing one GNN controller."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlqls.features.specs import AtomGraphSpec, PulseLibrarySpec, SpectroscopyGraphSpec
from rlqls.model import BranchModel


@dataclass(slots=True)
class MoleculeTask:
    """One molecule/configuration-specific MDP plus transferable descriptors.

    The quantum transition law remains molecule-specific through
    ``model.branch_matrices``.  Only the neural parameters are shared.
    """

    task_id: int
    name: str
    family: str
    model: BranchModel
    atom_graph: AtomGraphSpec
    spectroscopy_graph: SpectroscopyGraphSpec
    pulse_library: PulseLibrarySpec
    experimental_features: np.ndarray
    transfer_group: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.experimental_features = np.asarray(
            self.experimental_features, dtype=np.float32
        )
        if self.experimental_features.ndim != 1:
            raise ValueError("experimental_features must be one-dimensional")
        if self.spectroscopy_graph.n_nodes != self.model.n_states:
            raise ValueError("spectroscopy graph/state count mismatch")
        if self.pulse_library.n_actions != self.model.n_actions:
            raise ValueError("pulse library/action count mismatch")
        if self.pulse_library.transition_src.size:
            if np.max(self.pulse_library.transition_src) >= self.model.n_states:
                raise ValueError("pulse transition source out of range")
            if np.max(self.pulse_library.transition_dst) >= self.model.n_states:
                raise ValueError("pulse transition destination out of range")


    def maximum_one_step_conditional_purity(self) -> float:
        """Upper diagnostic over basis-state inputs and measurement branches.

        For a basis population ``e_i``, pulse ``a``, and outcome ``k``, the
        conditional population is ``B[a,k,:,i] / sum_j B[a,k,j,i]``.  The
        returned number is the largest component over all such columns.  It is
        not a reachability proof, but it detects surrogate noise floors that
        make a requested terminal confidence impossible even in one branch.
        """

        matrices = np.asarray(self.model.branch_matrices, dtype=np.float64)
        mass = matrices.sum(axis=2, keepdims=True)
        normalized = np.divide(
            matrices,
            mass,
            out=np.zeros_like(matrices),
            where=mass > 0.0,
        )
        return float(np.max(normalized))

    @property
    def n_states(self) -> int:
        return self.model.n_states

    @property
    def n_actions(self) -> int:
        return self.model.n_actions


__all__ = ["MoleculeTask"]
