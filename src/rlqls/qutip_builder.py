"""Exact branch-matrix preprocessing using QuTiP.

Use this module when the complete molecular Hamiltonian and pulse library are
available.  It directly implements the physics-to-MDP reduction:

    U_a = T exp[-(i/hbar) integral H_a(t) dt],

    B[a,k]_{j i}
        = sum_{n in N_k} |<j,n| U_a |i,0>|^2.

The product basis is

    |i,n> = |i>_mol tensor |n>_mot,

where i labels a molecular internal eigenstate and n labels the selected shared
motional normal mode.  Coherent amplitudes are retained during propagation and
converted to population probabilities only after the pulse, when the motional
measurement branches are constructed.
"""
from __future__ import annotations

from collections.abc import Sequence

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
    """Propagate every ``|i,0>`` input and construct ``B[a,k]``.

    Parameters
    ----------
    molecular_dim:
        Number N of retained molecular eigenstates.
    motional_dim:
        Number of Fock states ``|n>`` retained for the selected shared mode.
    pulse_hamiltonians:
        One QuTiP Hamiltonian per action.  Each item may be a ``Qobj``,
        ``QobjEvo``, callable, or list-form time-dependent Hamiltonian accepted
        by ``qutip.sesolve``.
    pulse_durations_ms:
        Pulse duration for each action in the same time unit used by the
        Hamiltonian coefficients.
    outcome_zero, outcome_one:
        Sets of Fock numbers grouped into the binary detector results.  By
        default, k=0 means n=0 and k=1 means any retained n>=1.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(A,2,N,N)`` with column-stochastic total probability.
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

    branch_matrices = np.zeros(
        (len(pulse_hamiltonians), 2, molecular_dim, molecular_dim),
        dtype=np.float64,
    )
    options = {
        "atol": atol,
        "rtol": rtol,
        "store_states": True,
    }

    for action, (hamiltonian, duration_ms) in enumerate(
        zip(pulse_hamiltonians, pulse_durations_ms)
    ):
        for initial_molecular_state in range(molecular_dim):
            # Initial shared motion is recooled to |0> before every RL step.
            initial_joint_state = qt.tensor(
                qt.basis(molecular_dim, initial_molecular_state),
                qt.basis(motional_dim, 0),
            )

            # Closed-system Schrodinger propagation under the selected pulse.
            result = qt.sesolve(
                hamiltonian,
                initial_joint_state,
                [0.0, float(duration_ms)],
                e_ops=[],
                options=options,
            )
            final_vector = result.states[-1].full().reshape(
                molecular_dim,
                motional_dim,
            )

            # Project onto each coarse-grained motional outcome and sum over
            # unresolved Fock states.  The remaining axis labels final molecular
            # state j, so these probabilities form one input column i.
            for outcome, fock_numbers in enumerate(outcome_sets):
                branch_matrices[
                    action,
                    outcome,
                    :,
                    initial_molecular_state,
                ] = np.sum(
                    np.abs(final_vector[:, list(fock_numbers)]) ** 2,
                    axis=1,
                )

        # Numerical solvers may introduce tiny trace errors.  Normalize each
        # input column after summing over final molecule and measurement result.
        mass = branch_matrices[action].sum(axis=(0, 1))
        branch_matrices[action] /= mass[None, None, :]

    return branch_matrices.astype(np.float32)


__all__ = ["build_branch_matrices_qutip"]
