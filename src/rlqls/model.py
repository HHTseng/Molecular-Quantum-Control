"""Finite transition model for reinforcement-learning quantum-logic spectroscopy.

This module is the boundary between the molecular physics simulation and the
reinforcement-learning environment.

Notation
--------
Let N be the number of retained molecular eigenstates and A the number of pulse
choices.  The Gym observation is the molecular population column vector

    s = (P_1, ..., P_N)^T in Delta_{N-1},

where P_i is the posterior probability that the molecule occupies eigenstate
|i>.  The state is diagonal because the paper assumes that motional recooling
destroys the relevant coherences between successive control steps.

For action a and binary motional outcome k in {0, 1}, the precomputed
nonnegative matrix B[a, k] has entries

    B[a, k]_{j i}
        = sum_{n in outcome class k} |<j,n| U_a |i,0>|^2.

Thus, for an incoming population s,

    q_{a,k} = B[a,k] s                     (unnormalized branch population),
    p(k | s,a) = 1^T q_{a,k}              (measurement probability),
    F_{a,k}(s) = q_{a,k} / p(k | s,a)     (conditional next state).

The matrices combine three physical operations into one classical branch map:

1. coherent pulse evolution U_a on H_mol tensor H_mot;
2. projective measurement of the shared motional mode;
3. loss of inter-step coherence after motional cooling/reset.

The RL code never needs to solve the Schrodinger equation during training once
these branch matrices have been precomputed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_PLANCK = 6.626_070_15e-34
_BOLTZMANN = 1.380_649e-23


@dataclass(frozen=True, slots=True)
class BranchResult:
    """All measurement branches for one state-action pair.

    Attributes
    ----------
    probabilities:
        Shape ``(2,)``.  Entry k is p(k | s,a).
    next_states:
        Shape ``(2,N)``.  Row k is F_{a,k}(s).
    """

    probabilities: np.ndarray
    next_states: np.ndarray


@dataclass(slots=True)
class BranchModel:
    """Paper-level MDP transition model.

    ``branch_matrices[a, k, j, i]`` maps an input population at molecular state
    i to an output population at molecular state j, conditioned on motional
    outcome k.  Columns correspond to input basis states.  Therefore the
    physical trace-preservation condition is

        sum_{k,j} B[a,k]_{j i} = 1

    for every pulse a and every input state i.
    """

    branch_matrices: np.ndarray  # (A,2,N,N)
    initial_population: np.ndarray  # (N,)
    energies_over_h_khz: np.ndarray  # (N,)
    state_labels: list[str]
    action_labels: list[str]
    durations_ms: np.ndarray  # (A,)
    metadata: dict[str, Any] = field(default_factory=dict)
    bbr_propagators: np.ndarray | None = None  # (A,N,N), column stochastic

    def __post_init__(self) -> None:
        """Validate dimensions, positivity, and probability conservation."""

        self.branch_matrices = np.asarray(self.branch_matrices, dtype=np.float32)
        self.initial_population = normalize_probability(self.initial_population).astype(np.float32)
        self.energies_over_h_khz = np.asarray(self.energies_over_h_khz, dtype=np.float64)
        self.durations_ms = np.asarray(self.durations_ms, dtype=np.float64)

        if self.branch_matrices.ndim != 4 or self.branch_matrices.shape[1] != 2:
            raise ValueError("branch_matrices must have shape (A,2,N,N)")
        n_actions, _, n_out, n_in = self.branch_matrices.shape
        if n_out != n_in:
            raise ValueError("branch matrices must be square")
        if self.initial_population.shape != (n_in,):
            raise ValueError("initial_population shape mismatch")
        if self.energies_over_h_khz.shape != (n_in,):
            raise ValueError("energy shape mismatch")
        if len(self.state_labels) != n_in or len(self.action_labels) != n_actions:
            raise ValueError("label count mismatch")
        if self.durations_ms.shape != (n_actions,):
            raise ValueError("duration count mismatch")
        if not np.all(np.isfinite(self.branch_matrices)) or np.min(self.branch_matrices) < -1e-7:
            raise ValueError("branch matrices must be finite and nonnegative")

        self.branch_matrices = np.clip(self.branch_matrices, 0.0, None)

        # For a fixed input basis state i, probability may be distributed over
        # many final molecular states j and over both measurement outcomes k.
        # The sum of all such possibilities must be one.
        column_mass = self.branch_matrices.sum(axis=(1, 2))  # shape (A,N)
        error = float(np.max(np.abs(column_mass - 1.0)))
        if error > 2e-5:
            raise ValueError(f"branch maps are not trace preserving: {error:.3e}")
        self.branch_matrices /= column_mass[:, None, None, :]

        if self.bbr_propagators is not None:
            # BBR is a separate classical population map applied after the
            # pulse/measurement branch in this reconstruction.
            p = np.asarray(self.bbr_propagators, dtype=np.float32)
            if p.shape != (n_actions, n_in, n_in):
                raise ValueError("bbr_propagators must have shape (A,N,N)")
            if np.min(p) < -1e-7:
                raise ValueError("negative BBR matrix element")
            p = np.clip(p, 0.0, None)
            mass = p.sum(axis=1, keepdims=True)
            if np.min(mass) <= 0.0:
                raise ValueError("zero BBR column")
            self.bbr_propagators = p / mass

    @property
    def n_states(self) -> int:
        """Number N of molecular population components in the RL state."""

        return int(self.branch_matrices.shape[-1])

    @property
    def n_actions(self) -> int:
        """Number A of pulse choices in the discrete action library."""

        return int(self.branch_matrices.shape[0])

    @property
    def n_outcomes(self) -> int:
        """Binary coarse-grained motional readout: k=0 and k=1."""

        return 2

    def branches(self, state: np.ndarray, action: int, *, apply_bbr: bool = True) -> BranchResult:
        """Evaluate p(k|s,a) and F_{a,k}(s) for one state-action pair.

        This method does not sample the measurement.  It returns the complete
        stochastic transition kernel for the chosen pulse.  ``RLQLSEnv.step``
        later samples one branch, whereas the qMDP Bellman target averages over
        both branches analytically.
        """

        s = normalize_probability(np.asarray(state, dtype=np.float64))
        if not 0 <= int(action) < self.n_actions:
            raise ValueError("action out of range")

        maps = self.branch_matrices[int(action)].astype(np.float64, copy=False)

        # raw[k,j] = sum_i B[a,k]_{j i} s_i.
        raw = np.matmul(maps, s)  # (2,N)
        raw_mass = raw.sum(axis=1)  # p(k|s,a) before numerical normalization

        probabilities = np.clip(raw_mass, 0.0, None)
        probabilities /= probabilities.sum()

        # Conditional Bayesian/post-measurement states.  A zero-probability
        # branch is assigned the input state only to avoid NaNs; it contributes
        # zero weight to every exact expectation.
        next_states = np.empty_like(raw)
        for k in range(2):
            next_states[k] = raw[k] / raw_mass[k] if raw_mass[k] > 1e-15 else s

        if apply_bbr and self.bbr_propagators is not None:
            noise = self.bbr_propagators[int(action)]
            next_states = np.matmul(noise, next_states.T).T
            next_states = np.clip(next_states, 0.0, None)
            next_states /= next_states.sum(axis=1, keepdims=True)

        # Enforce exact binary normalization after floating-point arithmetic.
        probabilities[1] = 1.0 - probabilities[0]
        return BranchResult(probabilities, next_states)

    def batch_branches(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        apply_bbr: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized version of :meth:`branches` for replay batches.

        Parameters
        ----------
        states:
            Array of shape ``(batch,N)``.
        actions:
            Integer pulse indices of shape ``(batch,)``.

        Returns
        -------
        probabilities:
            Shape ``(batch,2)``.
        next_states:
            Shape ``(batch,2,N)``.
        """

        s = np.asarray(states, dtype=np.float64)
        a = np.asarray(actions, dtype=np.int64)
        if s.ndim != 2 or s.shape[1] != self.n_states:
            raise ValueError("states must have shape (batch,N)")
        if a.shape != (s.shape[0],):
            raise ValueError("actions must have shape (batch,)")
        if np.any((a < 0) | (a >= self.n_actions)):
            raise ValueError("action out of range")
        if np.min(s) < -1e-10 or not np.all(np.isfinite(s)):
            raise ValueError("invalid population")

        s = np.clip(s, 0.0, None)
        s /= s.sum(axis=1, keepdims=True)

        maps = self.branch_matrices[a].astype(np.float64, copy=False)  # (B,2,N,N)
        raw = np.einsum("bkij,bj->bki", maps, s, optimize=True)
        raw_mass = raw.sum(axis=2)
        probabilities = np.clip(raw_mass, 0.0, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        next_states = np.empty_like(raw)
        for k in range(2):
            mask = raw_mass[:, k] > 1e-15
            next_states[mask, k] = raw[mask, k] / raw_mass[mask, k, None]
            next_states[~mask, k] = s[~mask]

        if apply_bbr and self.bbr_propagators is not None:
            noise = self.bbr_propagators[a].astype(np.float64, copy=False)
            next_states = np.einsum("bij,bkj->bki", noise, next_states, optimize=True)
            next_states = np.clip(next_states, 0.0, None)
            next_states /= next_states.sum(axis=2, keepdims=True)

        probabilities[:, 1] = 1.0 - probabilities[:, 0]
        return probabilities, next_states

    def save_npz(self, path: str | Path) -> Path:
        """Serialize the complete finite MDP physics model."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            branch_matrices=self.branch_matrices,
            initial_population=self.initial_population,
            energies_over_h_khz=self.energies_over_h_khz,
            state_labels=np.asarray(self.state_labels, dtype=object),
            action_labels=np.asarray(self.action_labels, dtype=object),
            durations_ms=self.durations_ms,
            metadata=np.asarray([self.metadata], dtype=object),
            bbr_propagators=(
                self.bbr_propagators
                if self.bbr_propagators is not None
                else np.asarray([], dtype=np.float32)
            ),
        )
        return path

    @classmethod
    def load_npz(cls, path: str | Path) -> "BranchModel":
        """Load a serialized finite MDP physics model."""

        with np.load(path, allow_pickle=True) as d:
            bbr = d["bbr_propagators"]
            return cls(
                branch_matrices=d["branch_matrices"],
                initial_population=d["initial_population"],
                energies_over_h_khz=d["energies_over_h_khz"],
                state_labels=list(d["state_labels"].tolist()),
                action_labels=list(d["action_labels"].tolist()),
                durations_ms=d["durations_ms"],
                metadata=dict(d["metadata"].item()),
                bbr_propagators=None if bbr.size == 0 else bbr,
            )


def normalize_probability(vector: np.ndarray) -> np.ndarray:
    """Project a finite nonnegative vector onto the probability simplex."""

    x = np.asarray(vector, dtype=np.float64)
    if x.ndim != 1 or np.any(x < 0.0) or not np.all(np.isfinite(x)):
        raise ValueError("expected finite nonnegative vector")
    mass = float(x.sum())
    if mass <= 0.0:
        raise ValueError("probability vector has zero mass")
    return x / mass


def boltzmann_population(energies_over_h_khz: np.ndarray, temperature_kelvin: float) -> np.ndarray:
    """Return P_i proportional to exp[-E_i/(k_B T)].

    The supplied energies are frequencies E_i/h in kHz.  Multiplication by
    Planck's constant and 1e3 converts them back to joules.
    """

    if temperature_kelvin <= 0.0:
        raise ValueError("temperature must be positive")
    nu_khz = np.asarray(energies_over_h_khz, dtype=np.float64)
    exponent = -(_PLANCK * 1e3 * nu_khz) / (_BOLTZMANN * temperature_kelvin)
    exponent -= np.max(exponent)  # stable softmax shift
    weights = np.exp(exponent)
    return weights / weights.sum()


def branch_matrices_from_directed_edges(
    n_states: int,
    actions: Sequence[Sequence[tuple[int, int, float]]],
    *,
    max_transfer_per_source: float = 1.0,
) -> np.ndarray:
    """Build a transparent population-only surrogate from directed pulses.

    Each tuple ``(source, target, probability)`` approximates a blue-sideband
    pulse that transfers population from ``|source,0>`` to ``|target,1>`` with
    the given probability.  The untransferred probability remains in
    ``|source,0>``.  Hence

        B[a,1]_{target,source} = probability,
        B[a,0]_{source,source} = 1 - total transferred probability.

    This constructor is used for the reduced CaH+ reconstruction.  It is not a
    replacement for coherent propagation when accurate Hamiltonians are known.
    """

    out = np.zeros((len(actions), 2, n_states, n_states), dtype=np.float64)
    for a, edges in enumerate(actions):
        out[a, 0] = np.eye(n_states)
        grouped: dict[int, list[tuple[int, float]]] = {}
        for source, target, probability in edges:
            if not (0 <= source < n_states and 0 <= target < n_states):
                raise ValueError("state index out of range")
            if not (0.0 <= probability <= 1.0):
                raise ValueError("probability out of range")
            grouped.setdefault(source, []).append((target, probability))

        for source, targets in grouped.items():
            total = sum(q for _, q in targets)
            scale = min(1.0, max_transfer_per_source / max(total, 1e-15))
            transferred = 0.0
            out[a, 0, :, source] = 0.0
            for target, q in targets:
                q_scaled = q * scale
                out[a, 1, target, source] += q_scaled
                transferred += q_scaled
            out[a, 0, source, source] = 1.0 - transferred

    return out.astype(np.float32)


__all__ = [
    "BranchModel",
    "BranchResult",
    "normalize_probability",
    "boltzmann_population",
    "branch_matrices_from_directed_edges",
]
