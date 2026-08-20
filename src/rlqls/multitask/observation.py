"""Padded graph observations for chemistry-conditioned multi-molecule RL-QLS.

This follows the size-aware QDX principle: padding supplies static tensor
shapes, while masks prevent padded nodes/edges/actions from affecting the
shared neural parameters.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np
import torch

from rlqls.features.spectroscopy import DYNAMIC_LEVEL_DIM, dynamic_level_features

from .padding import GraphPadding
from .registry import MoleculeTaskRegistry
from .task import MoleculeTask


GLOBAL_FEATURE_DIM = 14


class MolecularGraphObservation(NamedTuple):
    atom_features: torch.Tensor
    atom_edge_features: torch.Tensor
    atom_senders: torch.Tensor
    atom_receivers: torch.Tensor
    atom_mask: torch.Tensor
    atom_edge_mask: torch.Tensor

    explicit_chemistry_features: torch.Tensor
    explicit_chemistry_feature_mask: torch.Tensor

    level_features: torch.Tensor
    level_feature_mask: torch.Tensor
    level_edge_features: torch.Tensor
    level_senders: torch.Tensor
    level_receivers: torch.Tensor
    level_mask: torch.Tensor
    level_edge_mask: torch.Tensor

    global_features: torch.Tensor

    pulse_features: torch.Tensor
    pulse_feature_mask: torch.Tensor
    action_mask: torch.Tensor
    action_env_indices: torch.Tensor

    pulse_transition_action: torch.Tensor
    pulse_transition_src: torch.Tensor
    pulse_transition_dst: torch.Tensor
    pulse_transition_features: torch.Tensor
    pulse_transition_mask: torch.Tensor

    task_id: torch.Tensor


def obs_map(
    fn: Callable,
    *observations: MolecularGraphObservation,
) -> MolecularGraphObservation:
    return MolecularGraphObservation(
        *[fn(*leaves) for leaves in zip(*[tuple(obs) for obs in observations])]
    )


def obs_stack(
    observations: Sequence[MolecularGraphObservation],
) -> MolecularGraphObservation:
    if not observations:
        raise ValueError("cannot stack an empty observation sequence")
    return obs_map(lambda *leaves: torch.stack(leaves, dim=0), *observations)


def obs_to(
    observation: MolecularGraphObservation,
    device: torch.device | str,
) -> MolecularGraphObservation:
    return obs_map(lambda leaf: leaf.to(device), observation)


def observation_from_numpy(
    observation: dict[str, np.ndarray],
    *,
    device: torch.device | str | None = None,
) -> MolecularGraphObservation:
    tensors = MolecularGraphObservation(
        atom_features=torch.as_tensor(observation["atom_features"], dtype=torch.float32),
        atom_edge_features=torch.as_tensor(observation["atom_edge_features"], dtype=torch.float32),
        atom_senders=torch.as_tensor(observation["atom_senders"], dtype=torch.int64),
        atom_receivers=torch.as_tensor(observation["atom_receivers"], dtype=torch.int64),
        atom_mask=torch.as_tensor(observation["atom_mask"], dtype=torch.bool),
        atom_edge_mask=torch.as_tensor(observation["atom_edge_mask"], dtype=torch.bool),
        explicit_chemistry_features=torch.as_tensor(
            observation["explicit_chemistry_features"], dtype=torch.float32
        ),
        explicit_chemistry_feature_mask=torch.as_tensor(
            observation["explicit_chemistry_feature_mask"], dtype=torch.float32
        ),
        level_features=torch.as_tensor(observation["level_features"], dtype=torch.float32),
        level_feature_mask=torch.as_tensor(
            observation["level_feature_mask"], dtype=torch.float32
        ),
        level_edge_features=torch.as_tensor(
            observation["level_edge_features"], dtype=torch.float32
        ),
        level_senders=torch.as_tensor(observation["level_senders"], dtype=torch.int64),
        level_receivers=torch.as_tensor(observation["level_receivers"], dtype=torch.int64),
        level_mask=torch.as_tensor(observation["level_mask"], dtype=torch.bool),
        level_edge_mask=torch.as_tensor(observation["level_edge_mask"], dtype=torch.bool),
        global_features=torch.as_tensor(observation["global_features"], dtype=torch.float32),
        pulse_features=torch.as_tensor(observation["pulse_features"], dtype=torch.float32),
        pulse_feature_mask=torch.as_tensor(
            observation["pulse_feature_mask"], dtype=torch.float32
        ),
        action_mask=torch.as_tensor(observation["action_mask"], dtype=torch.bool),
        action_env_indices=torch.as_tensor(
            observation["action_env_indices"], dtype=torch.int64
        ),
        pulse_transition_action=torch.as_tensor(
            observation["pulse_transition_action"], dtype=torch.int64
        ),
        pulse_transition_src=torch.as_tensor(
            observation["pulse_transition_src"], dtype=torch.int64
        ),
        pulse_transition_dst=torch.as_tensor(
            observation["pulse_transition_dst"], dtype=torch.int64
        ),
        pulse_transition_features=torch.as_tensor(
            observation["pulse_transition_features"], dtype=torch.float32
        ),
        pulse_transition_mask=torch.as_tensor(
            observation["pulse_transition_mask"], dtype=torch.bool
        ),
        task_id=torch.as_tensor(observation["task_id"], dtype=torch.int64),
    )
    return tensors if device is None else obs_to(tensors, device)


class MolecularGraphObservationBuilder:
    """Cache static task graphs and insert the current population dynamically."""

    def __init__(
        self,
        registry: MoleculeTaskRegistry,
        padding: GraphPadding | None = None,
    ) -> None:
        self.registry = registry
        self.padding = padding or GraphPadding.from_registry(registry)
        self._static = {
            task.name: self._build_static(task) for task in registry.tasks
        }

    @property
    def level_input_dim(self) -> int:
        return DYNAMIC_LEVEL_DIM + self.registry.feature_dimensions["level_static"]

    def _build_static(self, task: MoleculeTask) -> dict[str, np.ndarray]:
        p = self.padding
        if task.atom_graph.n_nodes > p.atoms_max:
            raise ValueError("atom graph exceeds padding")
        if task.atom_graph.n_edges > p.atom_edges_max:
            raise ValueError("atom edge graph exceeds padding")
        if task.n_states > p.states_max:
            raise ValueError("level graph exceeds padding")
        if task.spectroscopy_graph.n_edges > p.level_edges_max:
            raise ValueError("level edge graph exceeds padding")
        if task.n_actions > p.actions_max:
            raise ValueError("action list exceeds padding")
        if task.pulse_library.n_transitions > p.pulse_transitions_max:
            raise ValueError("pulse-transition list exceeds padding")

        dims = self.registry.feature_dimensions
        out: dict[str, np.ndarray] = {}

        out["atom_features"] = np.zeros(
            (p.atoms_max, dims["atom"]), dtype=np.float32
        )
        out["atom_features"][: task.atom_graph.n_nodes] = task.atom_graph.node_features
        out["atom_mask"] = np.zeros(p.atoms_max, dtype=np.int8)
        out["atom_mask"][: task.atom_graph.n_nodes] = 1
        out["atom_edge_features"] = np.zeros(
            (p.atom_edges_max, dims["atom_edge"]), dtype=np.float32
        )
        out["atom_edge_features"][: task.atom_graph.n_edges] = task.atom_graph.edge_features
        out["atom_senders"] = np.zeros(p.atom_edges_max, dtype=np.int64)
        out["atom_receivers"] = np.zeros(p.atom_edges_max, dtype=np.int64)
        out["atom_senders"][: task.atom_graph.n_edges] = task.atom_graph.senders
        out["atom_receivers"][: task.atom_graph.n_edges] = task.atom_graph.receivers
        out["atom_edge_mask"] = np.zeros(p.atom_edges_max, dtype=np.int8)
        out["atom_edge_mask"][: task.atom_graph.n_edges] = 1
        out["explicit_chemistry_features"] = task.atom_graph.explicit_features.copy()
        out["explicit_chemistry_feature_mask"] = (
            task.atom_graph.explicit_feature_mask.copy()
        )

        out["level_static_features"] = np.zeros(
            (p.states_max, dims["level_static"]), dtype=np.float32
        )
        out["level_static_features"][: task.n_states] = (
            task.spectroscopy_graph.static_node_features
        )
        out["level_static_feature_mask"] = np.zeros_like(
            out["level_static_features"]
        )
        out["level_static_feature_mask"][: task.n_states] = (
            task.spectroscopy_graph.static_node_feature_mask
        )
        out["level_mask"] = np.zeros(p.states_max, dtype=np.int8)
        out["level_mask"][: task.n_states] = 1
        out["level_edge_features"] = np.zeros(
            (p.level_edges_max, dims["level_edge"]), dtype=np.float32
        )
        out["level_edge_features"][: task.spectroscopy_graph.n_edges] = (
            task.spectroscopy_graph.edge_features
        )
        out["level_senders"] = np.zeros(p.level_edges_max, dtype=np.int64)
        out["level_receivers"] = np.zeros(p.level_edges_max, dtype=np.int64)
        out["level_senders"][: task.spectroscopy_graph.n_edges] = (
            task.spectroscopy_graph.senders
        )
        out["level_receivers"][: task.spectroscopy_graph.n_edges] = (
            task.spectroscopy_graph.receivers
        )
        out["level_edge_mask"] = np.zeros(p.level_edges_max, dtype=np.int8)
        out["level_edge_mask"][: task.spectroscopy_graph.n_edges] = 1

        out["pulse_features"] = np.zeros(
            (p.actions_max, dims["pulse"]), dtype=np.float32
        )
        out["pulse_feature_mask"] = np.zeros_like(out["pulse_features"])
        out["pulse_features"][: task.n_actions] = task.pulse_library.pulse_features
        out["pulse_feature_mask"][: task.n_actions] = (
            task.pulse_library.pulse_feature_mask
        )
        out["action_mask"] = np.zeros(p.actions_max, dtype=np.int8)
        out["action_mask"][: task.n_actions] = 1
        out["action_env_indices"] = np.full(p.actions_max, -1, dtype=np.int64)
        out["action_env_indices"][: task.n_actions] = (
            task.pulse_library.action_env_indices
        )

        t_count = task.pulse_library.n_transitions
        out["pulse_transition_action"] = np.zeros(
            p.pulse_transitions_max, dtype=np.int64
        )
        out["pulse_transition_src"] = np.zeros(
            p.pulse_transitions_max, dtype=np.int64
        )
        out["pulse_transition_dst"] = np.zeros(
            p.pulse_transitions_max, dtype=np.int64
        )
        out["pulse_transition_features"] = np.zeros(
            (p.pulse_transitions_max, dims["pulse_transition"]), dtype=np.float32
        )
        out["pulse_transition_mask"] = np.zeros(
            p.pulse_transitions_max, dtype=np.int8
        )
        out["pulse_transition_action"][:t_count] = (
            task.pulse_library.transition_action
        )
        out["pulse_transition_src"][:t_count] = task.pulse_library.transition_src
        out["pulse_transition_dst"][:t_count] = task.pulse_library.transition_dst
        out["pulse_transition_features"][:t_count] = (
            task.pulse_library.transition_features
        )
        out["pulse_transition_mask"][:t_count] = 1
        out["task_id"] = np.asarray(task.task_id, dtype=np.int64)
        return out

    def build(
        self,
        task: MoleculeTask,
        population: np.ndarray,
        *,
        step_count: int,
        max_steps: int,
    ) -> dict[str, np.ndarray]:
        static = self._static[task.name]
        state = np.asarray(population, dtype=np.float64)
        if state.shape != (task.n_states,):
            raise ValueError("population shape mismatch")
        state = np.clip(state, 0.0, None)
        state /= state.sum()

        dynamic = dynamic_level_features(state)
        level = np.zeros(
            (self.padding.states_max, self.level_input_dim), dtype=np.float32
        )
        level_mask = np.zeros_like(level)
        level[: task.n_states, :DYNAMIC_LEVEL_DIM] = dynamic
        level[: task.n_states, DYNAMIC_LEVEL_DIM:] = static["level_static_features"][
            : task.n_states
        ]
        level_mask[: task.n_states, :DYNAMIC_LEVEL_DIM] = 1.0
        level_mask[: task.n_states, DYNAMIC_LEVEL_DIM:] = static[
            "level_static_feature_mask"
        ][: task.n_states]

        entropy = -float(np.sum(state * np.log(state + 1e-12)))
        normalized_entropy = entropy / max(float(np.log(task.n_states)), 1e-12)
        confidence = float(np.max(state))
        step_fraction = float(step_count) / max(max_steps, 1)
        global_features = np.concatenate(
            [
                np.asarray(
                    [
                        normalized_entropy,
                        confidence,
                        step_fraction,
                        1.0 - step_fraction,
                        np.log1p(task.n_states) / 6.0,
                        np.log1p(task.n_actions) / 6.0,
                    ],
                    dtype=np.float32,
                ),
                task.experimental_features,
            ]
        )
        if global_features.shape != (GLOBAL_FEATURE_DIM,):
            raise RuntimeError("global feature dimension mismatch")

        observation = {
            key: value.copy()
            for key, value in static.items()
            if key not in {"level_static_features", "level_static_feature_mask"}
        }
        observation["level_features"] = level
        observation["level_feature_mask"] = level_mask
        observation["global_features"] = global_features.astype(np.float32)
        return observation


__all__ = [
    "GLOBAL_FEATURE_DIM",
    "MolecularGraphObservation",
    "MolecularGraphObservationBuilder",
    "observation_from_numpy",
    "obs_stack",
    "obs_to",
]
