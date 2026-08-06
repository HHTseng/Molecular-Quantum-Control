"""Optional exact branch-matrix preprocessing using QuTiP 5.x.

This implements the paper's physics-preprocessing prescription when the exact
molecular Hamiltonian and pulse library are supplied.  The material builders in
``materials.py`` do not call it because those unpublished inputs are absent.
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
import numpy as np


def build_branch_matrices_qutip(
    *,
    molecular_dim: int,
    motional_dim: int,
    pulse_hamiltonians: Sequence[object],
    pulse_durations_ms: Sequence[float],
    outcome_zero: Sequence[int] = (0,),
    outcome_one: Sequence[int] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> np.ndarray:
    """Solve ``U_a|i,0>`` and form the unnormalized branch maps (Sec. 3).

    Specifically, ``B[a,k,j,i]=sum_(n in N_k)|<j,n|U_a|i,0>|^2``.

    ``pulse_hamiltonians[a]`` may be a QuTiP Qobj, QobjEvo, or list-form
    time-dependent Hamiltonian accepted by ``qutip.sesolve``.
    """
    try:
        import qutip as qt
    except ImportError as exc:
        raise ImportError("Install optional dependency qutip>=5.3.0") from exc

    if outcome_one is None:
        outcome_one = tuple(range(1, motional_dim))
    outcome_sets = [tuple(outcome_zero), tuple(outcome_one)]

    if len(pulse_hamiltonians) != len(pulse_durations_ms):
        raise ValueError("pulse count mismatch")

    b = np.zeros((len(pulse_hamiltonians), 2, molecular_dim, molecular_dim), dtype=np.float64)
    options = {"atol": atol, "rtol": rtol, "store_states": True}

    for a, (hamiltonian, duration_ms) in enumerate(zip(pulse_hamiltonians, pulse_durations_ms)):
        for i in range(molecular_dim):

            # Start every coherent pulse in the cooled motional state |i> tensor |0>.
            psi0 = qt.tensor(qt.basis(molecular_dim, i), qt.basis(motional_dim, 0))
            result = qt.sesolve(hamiltonian, psi0, [0.0, float(duration_ms)], e_ops=[], options=options)

            # Reshape amplitudes <j,n|psi_f>; coherence is retained until this point.
            vector = result.states[-1].full().reshape(molecular_dim, motional_dim)

            for k, numbers in enumerate(outcome_sets):
                # Project onto N_k, discard phase, and sum unresolved motional outcomes.
                b[a, k, :, i] = np.sum(np.abs(vector[:, list(numbers)]) ** 2, axis=1)

        # Enforce sum_{k,j} B[a,k,j,i]=1 for every input basis state i.
        mass = b[a].sum(axis=(0, 1))
        b[a] /= mass[None, None, :]
    return b.astype(np.float32)


__all__ = ["build_branch_matrices_qutip"]
