"""Finite branch model for reinforcement-learning quantum-logic spectroscopy.

For a molecular population column vector ``s`` and pulse action ``a``, the
precomputed nonnegative matrices ``B[a,k]`` implement the paper's Eqs. (2)--(4):

    q_(a,k) = B[a,k] s,
    p(k | s,a) = 1^T q_(a,k),
    s'_(a,k) = q_(a,k) / p(k | s,a),       k in {0,1}.

The normalization in the last line is required by main-text Eq. (3) and
Supplemental Eq. (S3), even though main-text Eq. (4b) suppresses it in notation.
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
    """All measurement branches for one pulse (pseudocode Sec. 5).

    ``probabilities[..., k] = p_k(s,a)`` and
    ``next_states[..., k, :] = s'_{a,k}``, for outcomes ``k in {0,1}``.
    """

    probabilities: np.ndarray  # (..., 2): p_k(s,a)
    next_states: np.ndarray  # (..., 2, N_S): s'_{a,k}


@dataclass(slots=True)
class BranchModel:
    """Finite classical state model obtained from coherent pulse dynamics.

    Matrix columns label input molecular states ``i`` and rows label output
    states ``j``: ``B[a,k,j,i] = sum_{n in N_k}|<j,n|U_a|i,0>|^2``.
    See pseudocode Secs. 2.4 and 3.
    """

    branch_matrices: np.ndarray  # (N_A, 2, N_S, N_S): B_{a,k}
    initial_population: np.ndarray  # (N_S,): s_0 in the probability simplex
    energies_over_h_khz: np.ndarray  # (N_S,): E_i/h in kHz
    state_labels: list[str]
    action_labels: list[str]
    durations_ms: np.ndarray  # (N_A,): pulse durations tau_a
    metadata: dict[str, Any] = field(default_factory=dict)
    bbr_propagators: np.ndarray | None = None  # (N_A,N_S,N_S): T_BBR(tau_a)

    def __post_init__(self) -> None:
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

        # Trace preservation: sum_{k,j} B[a,k]_{j i}=1 for every input i.
        column_mass = self.branch_matrices.sum(axis=(1, 2))
        error = float(np.max(np.abs(column_mass - 1.0)))
        if error > 2e-5:
            raise ValueError(f"branch maps are not trace preserving: {error:.3e}")
        self.branch_matrices /= column_mass[:, None, None, :]

        if self.bbr_propagators is not None:
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
        return int(self.branch_matrices.shape[-1])

    @property
    def n_actions(self) -> int:
        return int(self.branch_matrices.shape[0])

    @property
    def n_outcomes(self) -> int:
        return 2

    def branches(self, state: np.ndarray, action: int, *, apply_bbr: bool = True) -> BranchResult:
        """Evaluate ``p_k=1^T B_{a,k}s`` and ``s'_{a,k}=B_{a,k}s/p_k``."""
        s = normalize_probability(np.asarray(state, dtype=np.float64))
        if not 0 <= int(action) < self.n_actions:
            raise ValueError("action out of range")

        # B stores probabilities |amplitude|^2, not quantum amplitudes.
        maps = self.branch_matrices[int(action)].astype(np.float64, copy=False)
        raw = np.matmul(maps, s)     # unnormalized branch: \tilde{s}_{a,k} = B_{a,k} s
        raw_mass = raw.sum(axis=1)   # Born probability p(k | s,a) = 1^T \tilde{s}_{a,k}
        probabilities = np.clip(raw_mass, 0.0, None)
        probabilities /= probabilities.sum()
        next_states = np.empty_like(raw)
        for k in range(2):
            next_states[k] = raw[k] / raw_mass[k] if raw_mass[k] > 1e-15 else s
        if apply_bbr and self.bbr_propagators is not None:
            # Sequential model: s' <- T_BBR(tau_a) s_cond (pseudocode Sec. 2.5).
            next_states = np.matmul(self.bbr_propagators[int(action)], next_states.T).T
            next_states = np.clip(next_states, 0.0, None)
            next_states /= next_states.sum(axis=1, keepdims=True)
        probabilities[1] = 1.0 - probabilities[0]
        return BranchResult(probabilities, next_states)

    def batch_branches(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        apply_bbr: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch form of :meth:`branches` for states ``s_b`` and actions ``a_b``."""
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
        raw = np.einsum("bkij,bj->bki", maps, s, optimize=True)  # B_{a_b,k} s_b
        raw_mass = raw.sum(axis=2)  # p_{b,k}=sum_i raw_{b,k,i}
        probabilities = np.clip(raw_mass, 0.0, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        next_states = np.empty_like(raw)
        for k in range(2):
            mask = raw_mass[:, k] > 1e-15
            next_states[mask, k] = raw[mask, k] / raw_mass[mask, k, None]
            # A zero-probability branch contributes zero to the qMDP expectation.
            next_states[~mask, k] = s[~mask]

        if apply_bbr and self.bbr_propagators is not None:
            noise = self.bbr_propagators[a].astype(np.float64, copy=False)
            next_states = np.einsum("bij,bkj->bki", noise, next_states, optimize=True)
            next_states = np.clip(next_states, 0.0, None)
            next_states /= next_states.sum(axis=2, keepdims=True)

        probabilities[:, 1] = 1.0 - probabilities[:, 0]
        return probabilities, next_states

    def save_npz(self, path: str | Path) -> Path:
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
    """Project a nonnegative vector onto ``Delta_(N_S-1)`` by ``x/(1^T x)``."""
    x = np.asarray(vector, dtype=np.float64)
    if x.ndim != 1 or np.any(x < 0.0) or not np.all(np.isfinite(x)):
        raise ValueError("expected finite nonnegative vector")
    mass = float(x.sum())
    if mass <= 0.0:
        raise ValueError("probability vector has zero mass")
    return x / mass


def Boltzmann_prob(energies_over_h_khz: np.ndarray, temperature_kelvin: float) -> np.ndarray:
    """Return ``s_0,i = exp[-E_i/(k_B T)] / Z`` with ``E_i=h*nu_i``."""
    if temperature_kelvin <= 0.0:
        raise ValueError("temperature must be positive")
    nu_khz = np.asarray(energies_over_h_khz, dtype=np.float64)
    exponent = -(_PLANCK * 1e3 * nu_khz) / (_BOLTZMANN * temperature_kelvin)
    exponent -= np.max(exponent)
    weights = np.exp(exponent)
    return weights / weights.sum()


def branch_matrices_from_directed_edges(
    n_states: int,
    actions: Sequence[Sequence[tuple[int, int, float]]],
    *,
    max_transfer_per_source: float = 1.0,
) -> np.ndarray:
    """Construct surrogate ``B_{a,k}`` from directed transfer probabilities.

    An edge ``(i,j,q)`` places probability ``q`` in the detected branch
    ``B[a,1,j,i]``; residual population remains in ``B[a,0,i,i]``.  This is a
    population-level surrogate, not a coherent construction of ``U_a``.
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
    "Boltzmann_prob",
    "branch_matrices_from_directed_edges",
]
