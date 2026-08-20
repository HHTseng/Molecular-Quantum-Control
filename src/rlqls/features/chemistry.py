"""Chemistry descriptors and atom/isotope graphs for transfer learning.

The chemistry graph is episode-static.  It conditions the shared control
network but is not itself the Markov state.  Isotope mass and nuclear spin are
explicit because isotopologues such as H3O+ and D3O+ have identical element
connectivity but measurably different rotational/inversion structure.

The small descriptor table below is sufficient for the included demonstration
molecules.  It is not intended as a replacement for a molecular database.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .specs import AtomGraphSpec


ATOM_FEATURE_DIM = 9
ATOM_EDGE_FEATURE_DIM = 5
EXPLICIT_CHEMISTRY_DIM = 16


@dataclass(frozen=True, slots=True)
class Isotope:
    label: str
    atomic_number: int
    mass_u: float
    nuclear_spin: float
    electronegativity: float
    is_metal: bool = False


ISOTOPES: dict[str, Isotope] = {
    "H": Isotope("1H", 1, 1.00782503223, 0.5, 2.20, False),
    "D": Isotope("2H", 1, 2.01410177812, 1.0, 2.20, False),
    "O": Isotope("16O", 8, 15.99491461957, 0.0, 3.44, False),
    "Mg": Isotope("24Mg", 12, 23.985041697, 0.0, 1.31, True),
    "Ca": Isotope("40Ca", 20, 39.962590863, 0.0, 1.00, True),
}


def _atom_feature(isotope: Isotope) -> np.ndarray:
    """Dimensionless atom feature vector used by the chemistry MPNN."""

    z = float(isotope.atomic_number)
    mass = float(isotope.mass_u)
    return np.asarray(
        [
            z / 100.0,
            mass / 200.0,
            isotope.nuclear_spin / 5.0,
            isotope.electronegativity / 4.0,
            float(z == 1),
            float(z == 8),
            float(isotope.is_metal),
            float(mass > 1.5 and z == 1),  # deuterium/isotopic-H flag
            1.0,
        ],
        dtype=np.float32,
    )


def _edge_feature(
    left: Isotope,
    right: Isotope,
    *,
    bond_order: float,
    distance_angstrom: float,
) -> np.ndarray:
    mass_ratio = min(left.mass_u, right.mass_u) / max(left.mass_u, right.mass_u)
    return np.asarray(
        [
            bond_order / 3.0,
            distance_angstrom / 4.0,
            mass_ratio,
            float(left.atomic_number == right.atomic_number),
            1.0,
        ],
        dtype=np.float32,
    )


def build_atom_graph(
    atom_symbols: Sequence[str],
    undirected_bonds: Sequence[tuple[int, int, float, float]],
    *,
    total_charge: int,
    multiplicity: int = 1,
    symmetry_code: tuple[float, float, float] = (0.0, 0.0, 0.0),
    explicit_rotational_constants_ghz: Sequence[float] | None = None,
) -> AtomGraphSpec:
    """Create a directed atom graph and transparent global descriptors.

    Parameters
    ----------
    undirected_bonds:
        Tuples ``(i,j,bond_order,distance_angstrom)``.  Every bond is inserted
        in both directions so message passing remains orientation agnostic.
    symmetry_code:
        Three generic structural flags used only for the demonstration:
        ``(diatomic, trigonal_pyramidal, isotopically_symmetric)``.
    """

    isotopes = [ISOTOPES[symbol] for symbol in atom_symbols]
    node_features = np.stack([_atom_feature(atom) for atom in isotopes])

    senders: list[int] = []
    receivers: list[int] = []
    edge_features: list[np.ndarray] = []
    for i, j, bond_order, distance in undirected_bonds:
        if not (0 <= i < len(isotopes) and 0 <= j < len(isotopes)):
            raise ValueError("bond endpoint out of range")
        for source, target in ((i, j), (j, i)):
            senders.append(source)
            receivers.append(target)
            edge_features.append(
                _edge_feature(
                    isotopes[source],
                    isotopes[target],
                    bond_order=bond_order,
                    distance_angstrom=distance,
                )
            )

    if edge_features:
        edge_array = np.stack(edge_features).astype(np.float32)
    else:
        edge_array = np.zeros((0, ATOM_EDGE_FEATURE_DIM), dtype=np.float32)

    masses = np.asarray([atom.mass_u for atom in isotopes], dtype=np.float64)
    charges = np.asarray([atom.atomic_number for atom in isotopes], dtype=np.float64)
    spins = np.asarray([atom.nuclear_spin for atom in isotopes], dtype=np.float64)
    hydrogen_like = charges == 1
    heavy = charges > 1
    rotational = np.zeros(3, dtype=np.float32)
    rotational_mask = np.zeros(3, dtype=np.float32)
    if explicit_rotational_constants_ghz is not None:
        values = np.asarray(tuple(explicit_rotational_constants_ghz), dtype=np.float32)
        count = min(values.size, 3)
        rotational[:count] = np.log1p(np.maximum(values[:count], 0.0)) / 10.0
        rotational_mask[:count] = 1.0

    explicit = np.asarray(
        [
            total_charge / 5.0,
            len(isotopes) / 10.0,
            float(np.sum(masses)) / 300.0,
            float(np.mean(charges)) / 100.0,
            float(np.max(charges)) / 100.0,
            float(np.mean(hydrogen_like)),
            float(np.mean(heavy)),
            float(np.std(masses)) / 100.0,
            float(np.sum(spins)) / 10.0,
            multiplicity / 10.0,
            symmetry_code[0],
            symmetry_code[1],
            symmetry_code[2],
            rotational[0],
            rotational[1],
            rotational[2],
        ],
        dtype=np.float32,
    )
    explicit_mask = np.ones(EXPLICIT_CHEMISTRY_DIM, dtype=np.float32)
    explicit_mask[-3:] = rotational_mask

    return AtomGraphSpec(
        node_features=node_features,
        senders=np.asarray(senders, dtype=np.int64),
        receivers=np.asarray(receivers, dtype=np.int64),
        edge_features=edge_array,
        explicit_features=explicit,
        explicit_feature_mask=explicit_mask,
        atom_labels=tuple(atom.label for atom in isotopes),
    )


def build_cah_chemistry() -> AtomGraphSpec:
    # Representative geometry only; the branch-map physics does not depend on it.
    return build_atom_graph(
        ["Ca", "H"],
        [(0, 1, 1.0, 1.90)],
        total_charge=1,
        symmetry_code=(1.0, 0.0, 1.0),
    )


def build_mgh_chemistry() -> AtomGraphSpec:
    return build_atom_graph(
        ["Mg", "H"],
        [(0, 1, 1.0, 1.73)],
        total_charge=1,
        symmetry_code=(1.0, 0.0, 1.0),
    )


def build_h3o_chemistry(*, deuterated: bool = False) -> AtomGraphSpec:
    hydrogen = "D" if deuterated else "H"
    symbols = ["O", hydrogen, hydrogen, hydrogen]
    bonds = [(0, 1, 1.0, 0.98), (0, 2, 1.0, 0.98), (0, 3, 1.0, 0.98)]
    return build_atom_graph(
        symbols,
        bonds,
        total_charge=1,
        symmetry_code=(0.0, 1.0, 1.0),
    )


__all__ = [
    "ATOM_FEATURE_DIM",
    "ATOM_EDGE_FEATURE_DIM",
    "EXPLICIT_CHEMISTRY_DIM",
    "ISOTOPES",
    "build_atom_graph",
    "build_cah_chemistry",
    "build_mgh_chemistry",
    "build_h3o_chemistry",
]
