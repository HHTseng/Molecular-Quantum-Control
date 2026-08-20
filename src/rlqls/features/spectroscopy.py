"""Spectroscopy-state graph and candidate-pulse descriptor construction.

The current public reconstruction stores the exact finite MDP input as branch
maps ``B[a,k]`` but lacks all primitive pulse metadata.  This module therefore
supports a transparent ``branch_model`` descriptor mode:

    strength(a:i->j) = B[a,0]_{j i} + B[a,1]_{j i}.

That mode is appropriate for testing the variable-size GNN/control machinery,
but it exposes model-derived action effects.  A future high-fidelity database
should replace these records with primitive features such as Rabi amplitudes,
detunings, polarization, pulse duration, and selection-rule labels.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import re
from typing import Any

import numpy as np

from rlqls.model import BranchModel
from .specs import PulseLibrarySpec, SpectroscopyGraphSpec


LEVEL_STATIC_DIM = 14
LEVEL_EDGE_DIM = 11
PULSE_FEATURE_DIM = 14
PULSE_TRANSITION_DIM = 12
DYNAMIC_LEVEL_DIM = 3


def _empty_quantum_table(n_states: int) -> dict[str, np.ndarray]:
    nan = np.full(n_states, np.nan, dtype=np.float64)
    return {
        "J": nan.copy(),
        "K": nan.copy(),
        "mF": nan.copy(),
        "parity": nan.copy(),
        "xi": nan.copy(),
        "manifold": nan.copy(),
    }


def load_quantum_table(
    molecule_family: str,
    n_states: int,
    data_dir: str | Path,
) -> dict[str, np.ndarray]:
    """Load common quantum numbers for the included demonstration tasks."""

    data_dir = Path(data_dir)
    table = _empty_quantum_table(n_states)
    if molecule_family == "diatomic_hydride":
        path = data_dir / "cah16_states.csv"
        if not path.exists():
            return table
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != n_states:
            raise ValueError("CaH-like quantum table size mismatch")
        pattern = re.compile(r"J=([+-]?\d+),m=([+-]?\d+(?:/\d+)?),xi=([+-])")
        for row in rows:
            i = int(row["index"])
            match = pattern.search(row["description"])
            if match is None:
                continue
            table["J"][i] = float(match.group(1))
            m_text = match.group(2)
            if "/" in m_text:
                numerator, denominator = m_text.split("/")
                table["mF"][i] = float(numerator) / float(denominator)
            else:
                table["mF"][i] = float(m_text)
            table["parity"][i] = 1.0 if match.group(3) == "+" else -1.0
            table["xi"][i] = table["parity"][i]
        return table

    if molecule_family == "hydronium_isotopologue":
        path = data_dir / "h3o_states.csv"
        if not path.exists():
            return table
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != n_states:
            raise ValueError("H3O-like quantum table size mismatch")
        for row in rows:
            i = int(row["index"])
            table["J"][i] = float(row["J"])
            table["K"][i] = float(row["K"])
            table["mF"][i] = float(row["mF"])
            table["parity"][i] = 1.0 if row["parity"] == "+" else -1.0
            table["xi"][i] = float(row["xi"])
            table["manifold"][i] = float(row["manifold"])
        return table

    return table


def _normalize_energy(energies: np.ndarray) -> tuple[np.ndarray, float]:
    relative = np.asarray(energies, dtype=np.float64) - float(np.min(energies))
    positive = relative[relative > 0]
    scale = float(np.quantile(positive, 0.75)) if positive.size else 1.0
    scale = max(scale, 1.0)
    return relative, scale


def build_static_level_features(
    model: BranchModel,
    quantum: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    relative, scale = _normalize_energy(model.energies_over_h_khz)
    n = model.n_states
    feature = np.zeros((n, LEVEL_STATIC_DIM), dtype=np.float32)
    mask = np.ones_like(feature)

    feature[:, 0] = np.clip(relative / scale, 0.0, 20.0) / 20.0
    feature[:, 1] = np.log1p(relative) / max(float(np.log1p(np.max(relative))), 1.0)
    feature[:, 2] = np.arange(n, dtype=np.float32) / max(n - 1, 1)

    names = ["J", "K", "mF", "parity", "xi", "manifold"]
    normalizers = [6.0, 6.0, 8.0, 1.0, 6.0, 16.0]
    for offset, (name, normalizer) in enumerate(zip(names, normalizers), start=3):
        values = np.asarray(quantum[name], dtype=np.float64)
        available = np.isfinite(values)
        feature[available, offset] = values[available] / normalizer
        mask[~available, offset] = 0.0

    # Availability flags are explicit so the GNN can distinguish zero from absent.
    feature[:, 9] = np.isfinite(quantum["J"])
    feature[:, 10] = np.isfinite(quantum["K"])
    feature[:, 11] = np.isfinite(quantum["mF"])
    feature[:, 12] = np.isfinite(quantum["parity"])
    feature[:, 13] = np.isfinite(quantum["xi"])
    return feature, mask


def _parse_action_frequency(label: str) -> tuple[float, bool]:
    for pattern in (r"detuning=([+-]?\d+(?:\.\d+)?)", r"nu=([+-]?\d+(?:\.\d+)?)"):
        match = re.search(pattern, label)
        if match:
            return float(match.group(1)), True
    return 0.0, False


def _quantum_delta(
    quantum: dict[str, np.ndarray],
    name: str,
    source: int,
    target: int,
    scale: float,
) -> tuple[float, float]:
    left = quantum[name][source]
    right = quantum[name][target]
    if not (np.isfinite(left) and np.isfinite(right)):
        return 0.0, 0.0
    return float((right - left) / scale), 1.0


def build_pulse_library_from_branch_model(
    model: BranchModel,
    quantum: dict[str, np.ndarray],
    *,
    transition_threshold: float = 1e-2,
    max_transitions_per_action: int = 8,
) -> PulseLibrarySpec:
    """Construct sparse candidate-action records from ``B[a,k]``.

    For every non-negligible off-diagonal channel ``i -> j`` we retain both the
    no-phonon and phonon branch weights.  The local action index is used only to
    group records; it is not fed as a transferable feature.
    """

    b = np.asarray(model.branch_matrices, dtype=np.float64)
    n_actions = model.n_actions
    n_states = model.n_states
    relative, energy_scale = _normalize_energy(model.energies_over_h_khz)
    eta = float(model.metadata.get("lamb_dicke", 0.09))

    transition_action: list[int] = []
    transition_src: list[int] = []
    transition_dst: list[int] = []
    transition_features: list[np.ndarray] = []
    per_action_records: list[list[tuple[int, int, float, float]]] = []

    for action in range(n_actions):
        records: list[tuple[int, int, float, float]] = []
        total = b[action, 0] + b[action, 1]
        for source in range(n_states):
            for target in np.flatnonzero(total[:, source] > transition_threshold):
                target = int(target)
                if target == source:
                    continue
                b0 = float(b[action, 0, target, source])
                b1 = float(b[action, 1, target, source])
                strength = b0 + b1
                if strength <= transition_threshold:
                    continue
                records.append((source, target, b0, b1))
        # Dense surrogate cross-talk can create O(A N) tiny channels.  Candidate
        # scoring needs the dominant semantics, not every numerical tail.
        records.sort(key=lambda record: record[2] + record[3], reverse=True)
        if max_transitions_per_action > 0:
            records = records[:max_transitions_per_action]
        per_action_records.append(records)

    max_count = max((len(records) for records in per_action_records), default=1)
    pulse_features = np.zeros((n_actions, PULSE_FEATURE_DIM), dtype=np.float32)
    pulse_feature_mask = np.ones_like(pulse_features)

    for action, records in enumerate(per_action_records):
        frequency, frequency_present = _parse_action_frequency(model.action_labels[action])
        duration = float(model.durations_ms[action])
        totals = np.asarray([r[2] + r[3] for r in records], dtype=np.float64)
        b1_values = np.asarray([r[3] for r in records], dtype=np.float64)
        diagonal_noop = np.diag(b[action, 0])
        offdiag_targets_per_source = np.sum(
            (b[action].sum(axis=0) > transition_threshold), axis=0
        ) - (np.diag(b[action].sum(axis=0)) > transition_threshold)

        pulse_features[action] = np.asarray(
            [
                np.log1p(max(duration, 0.0)) / 5.0,
                np.sign(frequency) * np.log1p(abs(frequency)) / 25.0,
                eta,
                len(records) / max(max_count, 1),
                float(np.mean(totals)) if totals.size else 0.0,
                float(np.max(totals)) if totals.size else 0.0,
                float(np.mean(b1_values)) if b1_values.size else 0.0,
                float(np.max(b1_values)) if b1_values.size else 0.0,
                float(np.mean(diagonal_noop)),
                float(np.mean(np.maximum(offdiag_targets_per_source - 1, 0))) / max(n_states, 1),
                float(np.mean(np.sum(b[action, 1], axis=0))),
                float(frequency_present),
                0.0,  # pi polarization unavailable in branch-only reconstruction
                0.0,  # sigma polarization unavailable
            ],
            dtype=np.float32,
        )
        pulse_feature_mask[action, 1] = float(frequency_present)
        pulse_feature_mask[action, 12:14] = 0.0

        for source, target, b0, b1 in records:
            d_j, j_mask = _quantum_delta(quantum, "J", source, target, 6.0)
            d_k, k_mask = _quantum_delta(quantum, "K", source, target, 6.0)
            d_m, m_mask = _quantum_delta(quantum, "mF", source, target, 8.0)
            parity_i = quantum["parity"][source]
            parity_j = quantum["parity"][target]
            parity_same = (
                float(parity_i == parity_j)
                if np.isfinite(parity_i) and np.isfinite(parity_j)
                else 0.0
            )
            xi_i = quantum["xi"][source]
            xi_j = quantum["xi"][target]
            d_xi = (
                float((xi_j - xi_i) / 6.0)
                if np.isfinite(xi_i) and np.isfinite(xi_j)
                else 0.0
            )
            delta_e = float(
                (model.energies_over_h_khz[target] - model.energies_over_h_khz[source])
                / energy_scale
            )
            strength = b0 + b1
            transition_action.append(action)
            transition_src.append(source)
            transition_dst.append(target)
            transition_features.append(
                np.asarray(
                    [
                        b0,
                        b1,
                        strength,
                        b1 / max(strength, 1e-12),
                        np.clip(delta_e, -20.0, 20.0) / 20.0,
                        np.log1p(abs(delta_e)) / 5.0,
                        d_j,
                        d_k,
                        d_m,
                        parity_same,
                        d_xi,
                        1.0,  # branch-model descriptor mode flag
                    ],
                    dtype=np.float32,
                )
            )

    if transition_features:
        transition_array = np.stack(transition_features)
    else:
        transition_array = np.zeros((0, PULSE_TRANSITION_DIM), dtype=np.float32)

    return PulseLibrarySpec(
        pulse_features=pulse_features,
        pulse_feature_mask=pulse_feature_mask,
        transition_action=np.asarray(transition_action, dtype=np.int64),
        transition_src=np.asarray(transition_src, dtype=np.int64),
        transition_dst=np.asarray(transition_dst, dtype=np.int64),
        transition_features=transition_array,
        action_env_indices=np.arange(n_actions, dtype=np.int64),
        descriptor_mode="branch_model",
    )


def build_spectroscopy_graph(
    model: BranchModel,
    quantum: dict[str, np.ndarray],
    pulse_library: PulseLibrarySpec,
) -> SpectroscopyGraphSpec:
    static, static_mask = build_static_level_features(model, quantum)
    relative, energy_scale = _normalize_energy(model.energies_over_h_khz)

    grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(
        zip(pulse_library.transition_src, pulse_library.transition_dst)
    ):
        grouped[(int(source), int(target))].append(index)

    senders: list[int] = []
    receivers: list[int] = []
    features: list[np.ndarray] = []
    for (source, target), indices in grouped.items():
        transition = pulse_library.transition_features[indices]
        delta_e = float(
            (model.energies_over_h_khz[target] - model.energies_over_h_khz[source])
            / energy_scale
        )
        d_j, _ = _quantum_delta(quantum, "J", source, target, 6.0)
        d_k, _ = _quantum_delta(quantum, "K", source, target, 6.0)
        d_m, _ = _quantum_delta(quantum, "mF", source, target, 8.0)
        parity_i = quantum["parity"][source]
        parity_j = quantum["parity"][target]
        parity_same = (
            float(parity_i == parity_j)
            if np.isfinite(parity_i) and np.isfinite(parity_j)
            else 0.0
        )
        xi_i = quantum["xi"][source]
        xi_j = quantum["xi"][target]
        d_xi = (
            float((xi_j - xi_i) / 6.0)
            if np.isfinite(xi_i) and np.isfinite(xi_j)
            else 0.0
        )
        common = np.asarray(
            [
                np.log1p(abs(delta_e)) / 5.0,
                np.clip(delta_e, -20.0, 20.0) / 20.0,
                d_j,
                d_k,
                d_m,
                parity_same,
                d_xi,
                float(np.sum(transition[:, 2])),
                float(np.max(transition[:, 1])),
                len(indices) / max(model.n_actions, 1),
            ],
            dtype=np.float32,
        )
        # Include both directions.  The final feature marks the orientation.
        for left, right, direction in ((source, target, 1.0), (target, source, -1.0)):
            senders.append(left)
            receivers.append(right)
            oriented = common.copy()
            oriented[1:5] *= direction
            features.append(np.concatenate([oriented, np.asarray([direction], dtype=np.float32)]))

    if features:
        edge_array = np.stack(features).astype(np.float32)
    else:
        edge_array = np.zeros((0, LEVEL_EDGE_DIM), dtype=np.float32)

    return SpectroscopyGraphSpec(
        static_node_features=static,
        static_node_feature_mask=static_mask,
        senders=np.asarray(senders, dtype=np.int64),
        receivers=np.asarray(receivers, dtype=np.int64),
        edge_features=edge_array,
    )


def dynamic_level_features(population: np.ndarray) -> np.ndarray:
    """Return ``[P_i, log(P_i), -P_i log(P_i)]`` for every level node."""

    state = np.asarray(population, dtype=np.float64)
    state = np.clip(state, 0.0, None)
    state /= state.sum()
    eps = 1e-12
    return np.stack(
        [
            state,
            np.log(state + eps) / 30.0,
            -state * np.log(state + eps),
        ],
        axis=1,
    ).astype(np.float32)


__all__ = [
    "LEVEL_STATIC_DIM",
    "LEVEL_EDGE_DIM",
    "PULSE_FEATURE_DIM",
    "PULSE_TRANSITION_DIM",
    "DYNAMIC_LEVEL_DIM",
    "load_quantum_table",
    "build_static_level_features",
    "build_pulse_library_from_branch_model",
    "build_spectroscopy_graph",
    "dynamic_level_features",
]
