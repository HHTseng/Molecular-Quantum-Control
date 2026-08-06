"""Paper-informed CaH+ and H3O+ models.

The RL/control layer is paper-faithful.  The numerical physics layer is a
transparent reconstruction because the authors do not publish the full
pulse-conditioned transition matrices B[a,k].
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm

from .model import BranchModel, Boltzmann_prob, branch_matrices_from_directed_edges


@dataclass(frozen=True, slots=True)
class H3OTransition:
    """One tabulated Raman edge ``|i> <-> |f>`` with angular rate ``Omega_if``."""
    index: int
    source: int
    target: int
    frequency_khz: float
    rabi_over_2pi_khz: float
    Ji: int
    Ki: int
    parity_i: str
    mF_i: float
    xi_i: int
    Jf: int
    Kf: int
    parity_f: str
    mF_f: float
    xi_f: int


def CaH16_surrogate(T: float = 300.0) -> BranchModel:
    """Reconstruct the 16-state/13-action CaH+ example from Figs. 2(a), S2, S5.
       ``T`` is the temperature in ``s_0,i proportional exp[-E_i/(k_B T)]``.
    """
    labels = [
        "I: |1,-1/2,+>",
        "II: |1,+1/2,+>",
        "III: |1,+3/2,+>",
        "IV: |1,-3/2,->",
        "V: |1,-1/2,->",
        "VI: |1,+1/2,->",
        "VII: |2,-3/2,+>",
        "VIII: |2,-1/2,+>",
        "IX: |2,+1/2,+>",
        "X: |2,+3/2,+>",
        "XI: |2,+5/2,+>",
        "XII: |2,-5/2,->",
        "XIII: |2,-3/2,->",
        "XIV: |2,-1/2,->",
        "XV: |2,+1/2,->",
        "XVI: |2,+3/2,->",
    ]
    # Exact hyperfine energies are not tabulated.  The 0.57 THz rotational gap
    # dominates; figure-scale offsets preserve ordering but barely affect 300 K.
    energies = np.array(
        [0.0, 5.22, 10.44, 15.66, 20.88, 26.10]
        + [570_000_000.0 + x for x in [0, 4.18, 8.36, 12.53, 16.71, 20.89, 25.07, 29.24, 33.42, 37.60]], dtype=np.float64,)
    initial = Boltzmann_prob(energies, T)  # s_0 in Delta_15
    # Blue box -> red box direction in Supplemental Fig. S2.  Dominant labelled
    # efficiencies are retained; secondary efficiencies for pulses 3,4,9 are inferred.
    edges: list[list[tuple[int, int, float]]] = [
        [(10, 9, 0.89)],
        [(9, 8, 0.99)],
        [(8, 7, 0.94), (2, 1, 0.90)],
        [(7, 6, 0.85), (1, 0, 0.85)],
        [(6, 11, 1.00)],
        [(0, 5, 1.00)],
        [(15, 14, 0.99)],
        [(14, 13, 0.97)],
        [(13, 12, 0.74), (5, 4, 0.74)],
        [(3, 4, 1.00)],
        [(4, 3, 1.00)],
        [(11, 12, 1.00)],
        [(12, 11, 1.00)],
    ]
    # Convert inferred transfers i->j into B[a,1,j,i]; no-detection mass is B[a,0].
    matrices = branch_matrices_from_directed_edges(16, edges)
    frequencies = np.array([-1.72, -1.44, -1.03, -0.23, 4.40, 26.13, -6.12, -6.56, -7.33, 9.87, -9.87, 13.13, -13.13])
    table_D = np.array([16.2, 34.6, 52.6, 18.7, 28.5, 29.7, 16.6, 56.2, 23.7, 16.8, 16.8, 18.8, 18.8])

    return BranchModel(
        branch_matrices=matrices,
        initial_population=initial,
        energies_over_h_khz=energies,
        state_labels=labels,
        action_labels=[f"CaH+ pulse {a+1}: detuning={frequencies[a]:.2f} kHz" for a in range(13)],
        durations_ms=table_D / (2 * np.pi),
        metadata={
            "material": "CaH+",
            "paper_states": 16,
            "paper_actions": 13,
            "temperature_kelvin": T,
            "sweeping_order": list(range(13)),
            "model_kind": "dominant-edge surrogate",
            "uncertainties": [
                "Full QuTiP-derived 16x16 branch matrices are not published.",
                "Weak yellow/off-resonant channels in Fig. S2 are omitted.",
                "Secondary efficiencies for pulses 3, 4 and 9 are inferred.",
                "Exact hyperfine eigenenergies are approximated from figure spans.",
            ],
        },
    )


def _load_H3O(data_dir: Path) -> tuple[np.ndarray, list[str], list[H3OTransition]]:
    """Load molecular energies ``E_i/h`` and candidate Raman couplings."""
    energies: list[float] = []
    labels: list[str] = []
    with (data_dir / "H3O_states.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            energies.append(float(row["energy_over_h_khz"]))
            labels.append(
                f"|J={row['J']},K={row['K']},p={row['parity']},mF={row['mF']},xi={row['xi']}>"
            )
    transitions: list[H3OTransition] = []
    with (data_dir / "H3O_raman_transitions.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            transitions.append(
                H3OTransition(
                    index=int(row["transition_id"]),
                    source=int(row["source_index"]),
                    target=int(row["target_index"]),
                    frequency_khz=float(row["signed_frequency_khz"]),
                    rabi_over_2pi_khz=float(row["rabi_over_2pi_khz"]),
                    Ji=int(row["Ji"]),
                    Ki=int(row["Ki"]),
                    parity_i=row["parity_i"],
                    mF_i=float(row["mF_i"]),
                    xi_i=int(row["xi_i"]),
                    Jf=int(row["Jf"]),
                    Kf=int(row["Kf"]),
                    parity_f=row["parity_f"],
                    mF_f=float(row["mF_f"]),
                    xi_f=int(row["xi_f"]),
                )
            )
    if len(energies) != 130 or len(transitions) != 371:
        raise RuntimeError(
            f"expected paper tables 130/371; found {len(energies)}/{len(transitions)}"
        )
    return np.asarray(energies), labels, transitions


def _cluster(transitions: Iterable[H3OTransition], cutoff: float, tolerance: float):
    """Infer pulse actions by clustering near-degenerate transition frequencies."""
    ordered = sorted(
        (t for t in transitions if t.rabi_over_2pi_khz >= cutoff), key=lambda t: t.frequency_khz
    )
    clusters: list[list[H3OTransition]] = []
    for t in ordered:
        if not clusters or t.frequency_khz - clusters[-1][-1].frequency_khz > tolerance:
            clusters.append([t])
        else:
            clusters[-1].append(t)
    return clusters


class _UnionFind:
    """Find connected coherent subspaces of basis nodes ``(molecular state,n)``."""
    def __init__(self):
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx

    def components(self):
        groups: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
        for x in list(self.parent):
            groups[self.find(x)].add(x)
        return list(groups.values())


def _coherent_motional_map(
    n_states: int,
    energies: np.ndarray,
    pulse_frequency_khz: float,
    duration_ms: float,
    lamb_dicke: float,
    transitions: Sequence[H3OTransition],
    motional_dim: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Local coherent blue-sideband propagation with ``n=0,...,motional_dim-1``.

    The rotating-frame basis is ``|s,n>`` with diagonal angular frequency
    ``2*pi*(E_s/h - n*nu_a)`` (a common reference is removed).  Every tabulated
    molecular edge couples

        |i,n> <-> |f,n+1>

    with matrix element ``(eta*Omega_if/2)*sqrt(n+1)``.  Evolution is
    ``U_a=exp(-i H_a tau_a)``.  The final measurement
    branches are ``k=0`` for ``n=0`` and ``k=1`` for all ``n>=1``.
    """
    if motional_dim < 2:
        raise ValueError("motional_dim must be at least 2")
    b0 = np.eye(n_states, dtype=np.float64)
    b1 = np.zeros((n_states, n_states), dtype=np.float64)
    if not transitions:
        return b0, b1
    uf = _UnionFind()
    for t in transitions:
        for n in range(motional_dim - 1):
            uf.union((t.source, n), (t.target, n + 1))
    edges_by_root: dict[tuple[int, int], list[tuple[H3OTransition, int]]] = defaultdict(list)
    for t in transitions:
        for n in range(motional_dim - 1):
            edges_by_root[uf.find((t.source, n))].append((t, n))
    for component in uf.components():
        root = uf.find(next(iter(component)))
        edges = edges_by_root[root]
        nodes = sorted(component, key=lambda x: (x[1], x[0]))
        position = {node: p for p, node in enumerate(nodes)}
        # Rotating-frame Hamiltonian H_a/hbar in angular kHz; common energy is removed.
        h = np.zeros((len(nodes), len(nodes)), dtype=np.complex128)
        reference = float(energies[nodes[0][0]] - nodes[0][1] * pulse_frequency_khz)
        for (state, n), pos in position.items():
            h[pos, pos] = 2 * np.pi * (energies[state] - n * pulse_frequency_khz - reference)
        for t, n in edges:
            left = position[(t.source, n)]
            right = position[(t.target, n + 1)]
            # eta*Omega_if/2 * sqrt(n+1), with Omega_if=2*pi*(table rate).
            coupling = np.pi * lamb_dicke * t.rabi_over_2pi_khz * np.sqrt(n + 1.0)
            h[right, left] += coupling
            h[left, right] += coupling
        u = expm(-1j * h * duration_ms)  # U_a(tau_a)
        initial_sources = sorted(state for state, n in component if n == 0)
        for source in initial_sources:
            col = position[(source, 0)]
            b0[:, source] = 0.0
            b1[:, source] = 0.0
            for (dest, n), row in position.items():
                probability = abs(u[row, col]) ** 2  # |<dest,n|U_a|source,0>|^2
                if n == 0:
                    b0[dest, source] += probability
                else:
                    b1[dest, source] += probability
    b0 = np.clip(b0.real, 0, None)
    b1 = np.clip(b1.real, 0, None)
    # Trace preservation for each source i: sum_{k,j} B[k,j,i]=1.
    mass = b0.sum(axis=0) + b1.sum(axis=0)
    b0 /= mass[None, :]
    b1 /= mass[None, :]
    return b0, b1


def H3O_130_surrogate(
    data_dir: str | Path,
    *,
    temperature_kelvin: float = 20.0,
    lamb_dicke: float = 0.09,
    rate_cutoff_over_2pi_khz: float = 0.1,
    cluster_tolerance_khz: float = 0.43,
    drive_window_khz: float = 2.0,
    off_resonant_cutoff_over_2pi_khz: float = 0.05,
    motional_dim: int = 4,
    pulse_rule: str = "strongest",
) -> BranchModel:
    """Reconstruct a 130-state/218-action H3O+ model from Tables S3--S4.

    The 0.43 kHz adjacent-frequency clustering is selected because it yields
    218 groups.  It is not stated by the paper and is explicitly an inference.
    """
    data_dir = Path(data_dir)
    energies, labels, transitions = _load_H3O(data_dir)
    clusters = _cluster(transitions, rate_cutoff_over_2pi_khz, cluster_tolerance_khz)
    if len(clusters) != 218:
        raise RuntimeError(f"inferred action count {len(clusters)} != 218")
    initial = Boltzmann_prob(energies, temperature_kelvin)
    matrices = np.zeros((218, 2, 130, 130), dtype=np.float32)
    durations = []
    action_labels = []
    cluster_sizes = []
    local_sizes = []
    candidates = [t for t in transitions if t.rabi_over_2pi_khz >= off_resonant_cutoff_over_2pi_khz]
    for a, group in enumerate(clusters):
        if pulse_rule == "strongest":
            target = max(group, key=lambda t: t.rabi_over_2pi_khz)
            frequency = float(target.frequency_khz)
            # Blue-sideband pi time: tau_pi=pi/(eta*Omega)=1/(2 eta Omega/2pi).
            duration = float(1 / (2 * lamb_dicke * target.rabi_over_2pi_khz))
        elif pulse_rule == "cluster_median":
            frequency = float(np.mean([t.frequency_khz for t in group]))
            pi_times = [1 / (2 * lamb_dicke * t.rabi_over_2pi_khz) for t in group]
            duration = float(np.median(pi_times))
        else:
            raise ValueError("pulse_rule must be 'strongest' or 'cluster_median'")
        local = [t for t in candidates if abs(t.frequency_khz - frequency) <= drive_window_khz]
        b0, b1 = _coherent_motional_map(
            130, energies, frequency, duration, lamb_dicke, local, motional_dim=motional_dim
        )
        matrices[a, 0] = b0
        matrices[a, 1] = b1
        durations.append(duration)
        cluster_sizes.append(len(group))
        local_sizes.append(len(local))
        action_labels.append(f"H3O+ inferred pulse {a+1}: nu={frequency:.6f} kHz")
    return BranchModel(
        branch_matrices=matrices,
        initial_population=initial,
        energies_over_h_khz=energies,
        state_labels=labels,
        action_labels=action_labels,
        durations_ms=np.asarray(durations),
        metadata={
            "material": "H3O+",
            "paper_states": 130,
            "paper_actions": 218,
            "table_transitions": 371,
            "temperature_kelvin": temperature_kelvin,
            "lamb_dicke": lamb_dicke,
            "cluster_tolerance_khz": cluster_tolerance_khz,
            "drive_window_khz": drive_window_khz,
            "motional_dim": motional_dim,
            "pulse_rule": pulse_rule,
            "cluster_size_min_max": [min(cluster_sizes), max(cluster_sizes)],
            "local_edge_count_min_max": [min(local_sizes), max(local_sizes)],
            "sweeping_order": list(range(218)),
            "model_kind": "Table-S3/S4 local coherent surrogate",
            "uncertainties": [
                "Exact 218 pulse definitions and B[a,k] matrices are not published.",
                "218 actions are inferred using 0.43 kHz adjacent-frequency clustering.",
                "Cluster-to-pulse target selection is inferred; default uses the strongest transition.",
                "Four motional Fock states are retained and n>=1 outcomes are aggregated.",
                "The full H3O+ Hamiltonian/coupling derivation is deferred to a later work.",
                "Only transitions in a configurable local detuning window are propagated coherently.",
            ],
        },
    )


__all__ = ["CaH16_surrogate", "H3O_130_surrogate", "H3OTransition"]
