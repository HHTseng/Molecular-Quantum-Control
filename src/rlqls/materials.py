"""Paper-informed finite physics models for CaH+ and H3O+.

The RL and Gym layers operate only on branch matrices B[a,k].  This module
constructs approximate material-specific matrices from the information that is
available in the paper and supplement.

Important limitation
--------------------
The authors do not publish the exact pulse-conditioned transition matrices or
complete executable pulse library.  Therefore the builders below are explicit
surrogates:

* CaH+: a population-only directed-edge model reconstructed from Fig. S2.
* H3O+: a local coherent rotating-frame model reconstructed from Tables S3-S4.

Replace these builders with exact QuTiP-generated matrices when the full
Hamiltonian, pulse definitions, and experimental calibration are available.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm

from .model import (
    BranchModel,
    boltzmann_population,
    branch_matrices_from_directed_edges,
)


@dataclass(frozen=True, slots=True)
class H3OTransition:
    """One tabulated two-photon molecular Raman transition."""

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


def build_cah16_surrogate(temperature_kelvin: float = 300.0) -> BranchModel:
    """Build the 16-state, 13-pulse CaH+ reduced model.

    A directed edge ``i -> j`` represents the dominant blue-sideband process

        |i, n=0> -> |j, n=1>.

    The listed efficiency becomes ``B[a,1]_{j i}``; the remaining probability
    stays in ``|i,0>`` and contributes to ``B[a,0]_{i i}``.
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
    # dominates the 300 K Boltzmann weights.  Figure-scale offsets preserve the
    # ordering but are not a precision molecular Hamiltonian.
    energies = np.array(
        [0.0, 5.22, 10.44, 15.66, 20.88, 26.10]
        + [
            570_000_000.0 + x
            for x in [0, 4.18, 8.36, 12.53, 16.71, 20.89, 25.07, 29.24, 33.42, 37.60]
        ],
        dtype=np.float64,
    )
    initial = boltzmann_population(energies, temperature_kelvin)

    # Blue box -> red box direction in Supplemental Fig. S2.  A pulse may have
    # more than one edge when degenerate transitions are driven together.
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
    matrices = branch_matrices_from_directed_edges(16, edges)

    frequencies = np.array(
        [-1.72, -1.44, -1.03, -0.23, 4.40, 26.13, -6.12, -6.56, -7.33, 9.87, -9.87, 13.13, -13.13]
    )
    table_D = np.array(
        [16.2, 34.6, 52.6, 18.7, 28.5, 29.7, 16.6, 56.2, 23.7, 16.8, 16.8, 18.8, 18.8]
    )

    return BranchModel(
        branch_matrices=matrices,
        initial_population=initial,
        energies_over_h_khz=energies,
        state_labels=labels,
        action_labels=[
            f"CaH+ pulse {a + 1}: detuning={frequencies[a]:.2f} kHz"
            for a in range(13)
        ],
        durations_ms=table_D / (2 * np.pi),
        metadata={
            "material": "CaH+",
            "paper_states": 16,
            "paper_actions": 13,
            "temperature_kelvin": temperature_kelvin,
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


def _load_h3o(
    data_dir: Path,
) -> tuple[np.ndarray, list[str], list[H3OTransition]]:
    """Load the 130 energy levels and 371 tabulated Raman transitions."""

    energies: list[float] = []
    labels: list[str] = []
    with (data_dir / "h3o_states.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            energies.append(float(row["energy_over_h_khz"]))
            labels.append(
                f"|J={row['J']},K={row['K']},p={row['parity']},"
                f"mF={row['mF']},xi={row['xi']}>"
            )

    transitions: list[H3OTransition] = []
    with (data_dir / "h3o_raman_transitions.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        for row in csv.DictReader(file):
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


def _cluster(
    transitions: Iterable[H3OTransition],
    cutoff: float,
    tolerance: float,
) -> list[list[H3OTransition]]:
    """Infer pulse groups by adjacent transition-frequency clustering.

    This is a reconstruction heuristic, not a pulse-grouping rule stated by
    the paper.  The default tolerance was selected because it yields 218 groups,
    matching the reported H3O+ action count.
    """

    ordered = sorted(
        (t for t in transitions if t.rabi_over_2pi_khz >= cutoff),
        key=lambda t: t.frequency_khz,
    )
    clusters: list[list[H3OTransition]] = []
    for transition in ordered:
        if (
            not clusters
            or transition.frequency_khz - clusters[-1][-1].frequency_khz > tolerance
        ):
            clusters.append([transition])
        else:
            clusters[-1].append(transition)
    return clusters


class _UnionFind:
    """Group product states |i,n> connected by the same local pulse model."""

    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, x: tuple[int, int]) -> tuple[int, int]:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: tuple[int, int], y: tuple[int, int]) -> None:
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

    def components(self) -> list[set[tuple[int, int]]]:
        groups: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
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
    """Propagate a local blue-sideband Hamiltonian in the |i,n> basis.

    Product basis
    -------------
    ``(state,n)`` represents

        |state,n> = |state>_mol tensor |n>_mot.

    For every candidate molecular transition i -> f, the blue-sideband term
    couples

        |i,n> <-> |f,n+1>

    with matrix element

        (eta Omega_if / 2) sqrt(n+1).

    Since ``rabi_over_2pi_khz = Omega_if/(2 pi)`` and time is measured in ms,
    the angular-frequency matrix element used below is

        pi eta [Omega_if/(2 pi)] sqrt(n+1).

    Measurement coarse graining
    ----------------------------
    The final population is assigned to branch k=0 when n=0 and branch k=1
    when n>=1.  The code sums probabilities over unresolved motional states,
    matching the paper's binary readout approximation.
    """

    if motional_dim < 2:
        raise ValueError("motional_dim must be at least 2")

    # Default for molecular states untouched by the local pulse window:
    # remain in the n=0 branch.
    branch_zero = np.eye(n_states, dtype=np.float64)
    branch_one = np.zeros((n_states, n_states), dtype=np.float64)
    if not transitions:
        return branch_zero, branch_one

    # Split the Hamiltonian into connected blocks to avoid exponentiating the
    # full 130*motional_dim matrix for every inferred pulse.
    union_find = _UnionFind()
    for transition in transitions:
        for n in range(motional_dim - 1):
            union_find.union(
                (transition.source, n),
                (transition.target, n + 1),
            )

    edges_by_root: dict[
        tuple[int, int],
        list[tuple[H3OTransition, int]],
    ] = defaultdict(list)
    for transition in transitions:
        for n in range(motional_dim - 1):
            root = union_find.find((transition.source, n))
            edges_by_root[root].append((transition, n))

    for component in union_find.components():
        root = union_find.find(next(iter(component)))
        edges = edges_by_root[root]
        nodes = sorted(component, key=lambda x: (x[1], x[0]))
        position = {node: index for index, node in enumerate(nodes)}

        # Rotating-frame Hamiltonian in angular kHz.  The common reference only
        # removes a global phase and has no effect on transition probabilities.
        hamiltonian = np.zeros((len(nodes), len(nodes)), dtype=np.complex128)
        reference = float(
            energies[nodes[0][0]] - nodes[0][1] * pulse_frequency_khz
        )
        for (state, n), position_index in position.items():
            hamiltonian[position_index, position_index] = 2 * np.pi * (
                energies[state]
                - n * pulse_frequency_khz
                - reference
            )

        for transition, n in edges:
            left = position[(transition.source, n)]
            right = position[(transition.target, n + 1)]
            coupling = (
                np.pi
                * lamb_dicke
                * transition.rabi_over_2pi_khz
                * np.sqrt(n + 1.0)
            )
            hamiltonian[right, left] += coupling
            hamiltonian[left, right] += coupling

        # U_a = exp(-i H_a tau_a) for this time-independent rotating-frame
        # surrogate.  Exact time-dependent pulses would instead use QuTiP.
        unitary = expm(-1j * hamiltonian * duration_ms)

        initial_sources = sorted(state for state, n in component if n == 0)
        for source in initial_sources:
            initial_column = position[(source, 0)]
            branch_zero[:, source] = 0.0
            branch_one[:, source] = 0.0

            # Project U_a |source,0> onto all |dest,n>.  Phases are discarded
            # only after the coherent propagation, when constructing the
            # population-level measurement branches.
            for (destination, n), row in position.items():
                probability = abs(unitary[row, initial_column]) ** 2
                if n == 0:
                    branch_zero[destination, source] += probability
                else:
                    branch_one[destination, source] += probability

    branch_zero = np.clip(branch_zero.real, 0.0, None)
    branch_one = np.clip(branch_one.real, 0.0, None)
    mass = branch_zero.sum(axis=0) + branch_one.sum(axis=0)
    branch_zero /= mass[None, :]
    branch_one /= mass[None, :]
    return branch_zero, branch_one


def build_h3o130_surrogate(
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
    """Build the inferred 130-state, 218-pulse H3O+ surrogate.

    For each inferred pulse:

    1. select a pulse center frequency and approximate pi-pulse duration;
    2. collect nearby tabulated Raman transitions;
    3. exponentiate a coherent blue-sideband Hamiltonian in |i,n>;
    4. project final populations into k=0 and k=1 branch matrices.

    The 0.43 kHz clustering and pulse construction are assumptions required by
    missing public inputs, not claims about the authors' exact implementation.
    """

    data_dir = Path(data_dir)
    energies, labels, transitions = _load_h3o(data_dir)
    clusters = _cluster(
        transitions,
        rate_cutoff_over_2pi_khz,
        cluster_tolerance_khz,
    )
    if len(clusters) != 218:
        raise RuntimeError(f"inferred action count {len(clusters)} != 218")

    initial = boltzmann_population(energies, temperature_kelvin)
    matrices = np.zeros((218, 2, 130, 130), dtype=np.float32)
    durations: list[float] = []
    action_labels: list[str] = []
    cluster_sizes: list[int] = []
    local_sizes: list[int] = []

    candidates = [
        transition
        for transition in transitions
        if transition.rabi_over_2pi_khz >= off_resonant_cutoff_over_2pi_khz
    ]

    for action, group in enumerate(clusters):
        if pulse_rule == "strongest":
            target = max(group, key=lambda t: t.rabi_over_2pi_khz)
            frequency = float(target.frequency_khz)

            # For n=0, Omega_BSB = eta Omega.  A pi pulse obeys
            # eta Omega tau = pi, hence tau = 1/[2 eta Omega/(2 pi)].
            duration = float(
                1.0 / (2.0 * lamb_dicke * target.rabi_over_2pi_khz)
            )
        elif pulse_rule == "cluster_median":
            frequency = float(np.mean([t.frequency_khz for t in group]))
            pi_times = [
                1.0 / (2.0 * lamb_dicke * t.rabi_over_2pi_khz)
                for t in group
            ]
            duration = float(np.median(pi_times))
        else:
            raise ValueError("pulse_rule must be 'strongest' or 'cluster_median'")

        local = [
            transition
            for transition in candidates
            if abs(transition.frequency_khz - frequency) <= drive_window_khz
        ]
        branch_zero, branch_one = _coherent_motional_map(
            n_states=130,
            energies=energies,
            pulse_frequency_khz=frequency,
            duration_ms=duration,
            lamb_dicke=lamb_dicke,
            transitions=local,
            motional_dim=motional_dim,
        )

        matrices[action, 0] = branch_zero
        matrices[action, 1] = branch_one
        durations.append(duration)
        cluster_sizes.append(len(group))
        local_sizes.append(len(local))
        action_labels.append(
            f"H3O+ inferred pulse {action + 1}: nu={frequency:.6f} kHz"
        )

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


__all__ = ["build_cah16_surrogate", "build_h3o130_surrogate", "H3OTransition"]
