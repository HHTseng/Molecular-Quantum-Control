"""Extensible registry of molecule-specific RL-QLS tasks."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .task import MoleculeTask


class MoleculeTaskRegistry:
    def __init__(self, tasks: Iterable[MoleculeTask]) -> None:
        items = list(tasks)
        if not items:
            raise ValueError("registry requires at least one task")
        names = [task.name for task in items]
        ids = [task.task_id for task in items]
        if len(set(names)) != len(names) or len(set(ids)) != len(ids):
            raise ValueError("task names and ids must be unique")
        self.tasks = tuple(sorted(items, key=lambda task: task.task_id))
        self.by_name = {task.name: task for task in self.tasks}
        self.by_id = {task.task_id: task for task in self.tasks}

        self.max_atoms = max(task.atom_graph.n_nodes for task in self.tasks)
        self.max_atom_edges = max(task.atom_graph.n_edges for task in self.tasks)
        self.max_states = max(task.n_states for task in self.tasks)
        self.max_level_edges = max(
            task.spectroscopy_graph.n_edges for task in self.tasks
        )
        self.max_actions = max(task.n_actions for task in self.tasks)
        self.max_pulse_transitions = max(
            task.pulse_library.n_transitions for task in self.tasks
        )

        # All feature builders use common dimensions; validate that contract.
        first = self.tasks[0]
        dimensions = {
            "atom": first.atom_graph.node_features.shape[1],
            "atom_edge": first.atom_graph.edge_features.shape[1],
            "explicit": first.atom_graph.explicit_features.size,
            "level_static": first.spectroscopy_graph.static_node_features.shape[1],
            "level_edge": first.spectroscopy_graph.edge_features.shape[1],
            "pulse": first.pulse_library.pulse_features.shape[1],
            "pulse_transition": first.pulse_library.transition_features.shape[1],
            "experimental": first.experimental_features.size,
        }
        for task in self.tasks[1:]:
            observed = {
                "atom": task.atom_graph.node_features.shape[1],
                "atom_edge": task.atom_graph.edge_features.shape[1],
                "explicit": task.atom_graph.explicit_features.size,
                "level_static": task.spectroscopy_graph.static_node_features.shape[1],
                "level_edge": task.spectroscopy_graph.edge_features.shape[1],
                "pulse": task.pulse_library.pulse_features.shape[1],
                "pulse_transition": task.pulse_library.transition_features.shape[1],
                "experimental": task.experimental_features.size,
            }
            if observed != dimensions:
                raise ValueError(
                    f"feature dimensions differ for task {task.name}: {observed}"
                )
        self.feature_dimensions = dimensions

    def get(self, key: str | int | MoleculeTask) -> MoleculeTask:
        if isinstance(key, MoleculeTask):
            return key
        if isinstance(key, str):
            try:
                return self.by_name[key]
            except KeyError as exc:
                raise KeyError(f"unknown molecule task {key!r}") from exc
        try:
            return self.by_id[int(key)]
        except KeyError as exc:
            raise KeyError(f"unknown molecule task id {key!r}") from exc

    def sample(
        self,
        rng: np.random.Generator,
        probabilities: np.ndarray | None = None,
        allowed: tuple[str, ...] | None = None,
    ) -> MoleculeTask:
        tasks = self.tasks if allowed is None else tuple(self.by_name[name] for name in allowed)
        if not tasks:
            raise ValueError("allowed task set is empty")
        if probabilities is None:
            index = int(rng.integers(len(tasks)))
        else:
            p = np.asarray(probabilities, dtype=np.float64)
            if p.shape != (len(tasks),):
                raise ValueError("task probability shape mismatch")
            p = np.clip(p, 0.0, None)
            p /= p.sum()
            index = int(rng.choice(len(tasks), p=p))
        return tasks[index]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.tasks)


__all__ = ["MoleculeTaskRegistry"]
