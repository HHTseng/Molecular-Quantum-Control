"""Fixed-capacity graph buckets, following the size-aware QDX pattern."""
from __future__ import annotations

from dataclasses import dataclass

from .registry import MoleculeTaskRegistry


@dataclass(frozen=True, slots=True)
class GraphPadding:
    atoms_max: int
    atom_edges_max: int
    states_max: int
    level_edges_max: int
    actions_max: int
    pulse_transitions_max: int

    @classmethod
    def from_registry(
        cls,
        registry: MoleculeTaskRegistry,
        *,
        reserve_fraction: float = 0.0,
    ) -> "GraphPadding":
        if reserve_fraction < 0.0:
            raise ValueError("reserve_fraction must be nonnegative")

        def capacity(value: int) -> int:
            return max(value, int(round(value * (1.0 + reserve_fraction))))

        return cls(
            atoms_max=capacity(registry.max_atoms),
            atom_edges_max=capacity(registry.max_atom_edges),
            states_max=capacity(registry.max_states),
            level_edges_max=capacity(registry.max_level_edges),
            actions_max=capacity(registry.max_actions),
            pulse_transitions_max=capacity(registry.max_pulse_transitions),
        )


__all__ = ["GraphPadding"]
