"""Static graph and candidate-action records for multi-molecule RL-QLS.

The finite quantum-control physics remains in :class:`rlqls.model.BranchModel`.
These records expose transferable descriptors to a shared GNN:

* an atom/isotope graph for chemistry context;
* an internal-eigenstate graph for spectroscopy context;
* a variable-cardinality set of transitions for every candidate pulse.

No integer pulse index is treated as a universal semantic label.  A pulse is
identified through its descriptors and the transitions it drives.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AtomGraphSpec:
    node_features: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    edge_features: np.ndarray
    explicit_features: np.ndarray
    explicit_feature_mask: np.ndarray
    atom_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        node = np.asarray(self.node_features, dtype=np.float32)
        send = np.asarray(self.senders, dtype=np.int64)
        recv = np.asarray(self.receivers, dtype=np.int64)
        edge = np.asarray(self.edge_features, dtype=np.float32)
        explicit = np.asarray(self.explicit_features, dtype=np.float32)
        explicit_mask = np.asarray(self.explicit_feature_mask, dtype=np.float32)
        if node.ndim != 2:
            raise ValueError("atom node_features must be rank two")
        if send.shape != recv.shape or send.ndim != 1:
            raise ValueError("atom senders/receivers must have equal 1-D shape")
        if edge.shape != (send.size, edge.shape[-1] if edge.ndim == 2 else -1):
            raise ValueError("atom edge feature shape mismatch")
        if edge.ndim != 2:
            raise ValueError("atom edge_features must be rank two")
        if explicit.ndim != 1 or explicit_mask.shape != explicit.shape:
            raise ValueError("explicit chemistry feature shape mismatch")
        if len(self.atom_labels) != node.shape[0]:
            raise ValueError("atom label count mismatch")
        if send.size and (
            np.min(send) < 0
            or np.min(recv) < 0
            or np.max(send) >= node.shape[0]
            or np.max(recv) >= node.shape[0]
        ):
            raise ValueError("atom edge endpoint out of range")
        object.__setattr__(self, "node_features", node)
        object.__setattr__(self, "senders", send)
        object.__setattr__(self, "receivers", recv)
        object.__setattr__(self, "edge_features", edge)
        object.__setattr__(self, "explicit_features", explicit)
        object.__setattr__(self, "explicit_feature_mask", explicit_mask)

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.senders.size)


@dataclass(frozen=True, slots=True)
class SpectroscopyGraphSpec:
    static_node_features: np.ndarray
    static_node_feature_mask: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    edge_features: np.ndarray

    def __post_init__(self) -> None:
        node = np.asarray(self.static_node_features, dtype=np.float32)
        node_mask = np.asarray(self.static_node_feature_mask, dtype=np.float32)
        send = np.asarray(self.senders, dtype=np.int64)
        recv = np.asarray(self.receivers, dtype=np.int64)
        edge = np.asarray(self.edge_features, dtype=np.float32)
        if node.ndim != 2 or node_mask.shape != node.shape:
            raise ValueError("spectroscopy node feature shape mismatch")
        if send.shape != recv.shape or send.ndim != 1:
            raise ValueError("spectroscopy senders/receivers must be 1-D")
        if edge.ndim != 2 or edge.shape[0] != send.size:
            raise ValueError("spectroscopy edge feature shape mismatch")
        if send.size and (
            np.min(send) < 0
            or np.min(recv) < 0
            or np.max(send) >= node.shape[0]
            or np.max(recv) >= node.shape[0]
        ):
            raise ValueError("spectroscopy edge endpoint out of range")
        object.__setattr__(self, "static_node_features", node)
        object.__setattr__(self, "static_node_feature_mask", node_mask)
        object.__setattr__(self, "senders", send)
        object.__setattr__(self, "receivers", recv)
        object.__setattr__(self, "edge_features", edge)

    @property
    def n_nodes(self) -> int:
        return int(self.static_node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.senders.size)


@dataclass(frozen=True, slots=True)
class PulseLibrarySpec:
    pulse_features: np.ndarray
    pulse_feature_mask: np.ndarray
    transition_action: np.ndarray
    transition_src: np.ndarray
    transition_dst: np.ndarray
    transition_features: np.ndarray
    action_env_indices: np.ndarray
    descriptor_mode: str

    def __post_init__(self) -> None:
        pulse = np.asarray(self.pulse_features, dtype=np.float32)
        pulse_mask = np.asarray(self.pulse_feature_mask, dtype=np.float32)
        action = np.asarray(self.transition_action, dtype=np.int64)
        src = np.asarray(self.transition_src, dtype=np.int64)
        dst = np.asarray(self.transition_dst, dtype=np.int64)
        transition = np.asarray(self.transition_features, dtype=np.float32)
        env_indices = np.asarray(self.action_env_indices, dtype=np.int64)
        if pulse.ndim != 2 or pulse_mask.shape != pulse.shape:
            raise ValueError("pulse feature shape mismatch")
        if not (action.shape == src.shape == dst.shape) or action.ndim != 1:
            raise ValueError("pulse transition index arrays must be equal 1-D shapes")
        if transition.ndim != 2 or transition.shape[0] != action.size:
            raise ValueError("pulse transition feature shape mismatch")
        if env_indices.shape != (pulse.shape[0],):
            raise ValueError("action_env_indices shape mismatch")
        if action.size and (np.min(action) < 0 or np.max(action) >= pulse.shape[0]):
            raise ValueError("pulse transition action out of range")
        object.__setattr__(self, "pulse_features", pulse)
        object.__setattr__(self, "pulse_feature_mask", pulse_mask)
        object.__setattr__(self, "transition_action", action)
        object.__setattr__(self, "transition_src", src)
        object.__setattr__(self, "transition_dst", dst)
        object.__setattr__(self, "transition_features", transition)
        object.__setattr__(self, "action_env_indices", env_indices)

    @property
    def n_actions(self) -> int:
        return int(self.pulse_features.shape[0])

    @property
    def n_transitions(self) -> int:
        return int(self.transition_action.size)


__all__ = ["AtomGraphSpec", "SpectroscopyGraphSpec", "PulseLibrarySpec"]
