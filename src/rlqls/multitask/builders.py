"""Builders for source and related-transfer demonstration tasks.

Included tasks
--------------
* CaH+ and H3O+: the existing public reconstruction models.
* MgH+: a CaH+-topology related-task surrogate.
* D3O+: an H3O+-topology isotopologue surrogate.

MgH+ and D3O+ are deliberately marked as *transfer-demonstration surrogates*.
Their maps are deterministic deformations of the corresponding source model;
they are not claimed to be spectroscopically predictive.  This isolates and
tests the heterogeneous-policy/transfer machinery requested here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rlqls.features.chemistry import (
    build_cah_chemistry,
    build_h3o_chemistry,
    build_mgh_chemistry,
)
from rlqls.features.spectroscopy import (
    build_pulse_library_from_branch_model,
    build_spectroscopy_graph,
    load_quantum_table,
)
from rlqls.materials import build_cah16_surrogate, build_h3o130_surrogate
from rlqls.model import BranchModel, boltzmann_population

from .registry import MoleculeTaskRegistry
from .task import MoleculeTask


EXPERIMENTAL_FEATURE_DIM = 8


def _experimental_features(
    *,
    temperature_kelvin: float,
    magnetic_field_mt: float,
    motional_frequency_mhz: float,
    logic_ion_mass_u: float,
    infidelity_threshold: float,
    lamb_dicke: float,
    model_fidelity_flag: float,
    configuration_scale: float,
) -> np.ndarray:
    return np.asarray(
        [
            temperature_kelvin / 500.0,
            magnetic_field_mt / 10.0,
            motional_frequency_mhz / 10.0,
            logic_ion_mass_u / 200.0,
            infidelity_threshold / 0.1,
            lamb_dicke,
            model_fidelity_flag,
            configuration_scale,
        ],
        dtype=np.float32,
    )


def _neighbor_mixing(n_states: int, strength: float) -> np.ndarray:
    """Column-stochastic final-state cross-talk matrix."""

    if not 0.0 <= strength < 1.0:
        raise ValueError("cross-talk strength must lie in [0,1)")
    mixing = np.eye(n_states, dtype=np.float64) * (1.0 - strength)
    for source in range(n_states):
        neighbors = []
        if source > 0:
            neighbors.append(source - 1)
        if source + 1 < n_states:
            neighbors.append(source + 1)
        if neighbors:
            for target in neighbors:
                mixing[target, source] += strength / len(neighbors)
        else:
            mixing[source, source] += strength
    return mixing


def deform_related_branch_model(
    base: BranchModel,
    *,
    material: str,
    energy_scale: float,
    duration_scale: float,
    mean_pulse_retention: float,
    action_jitter: float,
    final_state_crosstalk: float,
    temperature_kelvin: float,
    seed: int,
    relation_note: str,
) -> BranchModel:
    r"""Create a close but non-identical task with preserved pulse semantics.

    For action ``a`` let ``rho_a`` be a deterministic retention factor.  The
    related map is

    .. math::

        \widetilde B_{a,k}
        = \rho_a B_{a,k}
          + (1-\rho_a)\,\delta_{k0} I,

    followed by a small column-stochastic final-state mixing ``T``:

    .. math:: B'_{a,k}=T\widetilde B_{a,k}.

    The deformation models changed pulse area and weak cross-talk while keeping
    the local action correspondence explicit.  It is an engineering benchmark,
    not a molecular Hamiltonian calculation.
    """

    rng = np.random.default_rng(seed)
    identity_zero = np.zeros_like(base.branch_matrices, dtype=np.float64)
    identity_zero[:, 0] = np.eye(base.n_states, dtype=np.float64)[None, :, :]
    retention = np.clip(
        mean_pulse_retention
        * np.exp(rng.normal(0.0, action_jitter, size=base.n_actions)),
        0.55,
        0.995,
    )
    b = base.branch_matrices.astype(np.float64)
    deformed = retention[:, None, None, None] * b + (
        1.0 - retention[:, None, None, None]
    ) * identity_zero

    mixing = _neighbor_mixing(base.n_states, final_state_crosstalk)
    deformed = np.einsum("ji,akil->akjl", mixing, deformed, optimize=True)
    deformed = np.clip(deformed, 0.0, None)
    mass = deformed.sum(axis=(1, 2))
    deformed /= mass[:, None, None, :]

    relative_energy = (
        base.energies_over_h_khz - float(np.min(base.energies_over_h_khz))
    ) * energy_scale
    initial = boltzmann_population(relative_energy, temperature_kelvin)
    metadata = dict(base.metadata)
    metadata.update(
        {
            "material": material,
            "temperature_kelvin": temperature_kelvin,
            "model_kind": "related-task transfer demonstration surrogate",
            "derived_from": base.metadata.get("material", "unknown"),
            "energy_scale": energy_scale,
            "duration_scale": duration_scale,
            "mean_pulse_retention": mean_pulse_retention,
            "action_jitter": action_jitter,
            "final_state_crosstalk": final_state_crosstalk,
            "relation_note": relation_note,
            "transfer_surrogate": True,
        }
    )
    return BranchModel(
        branch_matrices=deformed.astype(np.float32),
        initial_population=initial,
        energies_over_h_khz=relative_energy,
        state_labels=[label.replace(str(base.metadata.get("material", "")), material) for label in base.state_labels],
        action_labels=[
            label.replace(str(base.metadata.get("material", "")), material)
            for label in base.action_labels
        ],
        durations_ms=base.durations_ms * duration_scale,
        metadata=metadata,
    )


def _task_from_model(
    *,
    task_id: int,
    name: str,
    family: str,
    transfer_group: str,
    model: BranchModel,
    atom_graph,
    data_dir: Path,
    experimental_features: np.ndarray,
    metadata: dict | None = None,
) -> MoleculeTask:
    quantum = load_quantum_table(family, model.n_states, data_dir)
    pulses = build_pulse_library_from_branch_model(model, quantum)
    spectroscopy = build_spectroscopy_graph(model, quantum, pulses)
    return MoleculeTask(
        task_id=task_id,
        name=name,
        family=family,
        model=model,
        atom_graph=atom_graph,
        spectroscopy_graph=spectroscopy,
        pulse_library=pulses,
        experimental_features=experimental_features,
        transfer_group=transfer_group,
        metadata={} if metadata is None else dict(metadata),
    )


def build_default_registry(
    data_dir: str | Path,
    *,
    include_h3o: bool = True,
    use_precomputed_models: bool = True,
    precomputed_dir: str | Path | None = None,
) -> MoleculeTaskRegistry:
    """Build CaH+, H3O+, MgH+, and D3O+ demonstration tasks."""

    data_dir = Path(data_dir)
    precomputed = Path(precomputed_dir) if precomputed_dir is not None else None

    if use_precomputed_models and precomputed is not None and (
        precomputed / "cah16_surrogate_model.npz"
    ).exists():
        cah = BranchModel.load_npz(precomputed / "cah16_surrogate_model.npz")
    else:
        cah = build_cah16_surrogate()

    if include_h3o:
        if use_precomputed_models and precomputed is not None and (
            precomputed / "h3o130_surrogate_4mot_strongest.npz"
        ).exists():
            h3o = BranchModel.load_npz(
                precomputed / "h3o130_surrogate_4mot_strongest.npz"
            )
        else:
            h3o = build_h3o130_surrogate(data_dir)
    else:
        h3o = None

    # Reduced-mass-inspired energy scaling for a related diatomic benchmark.
    mass_ca = 39.962590863
    mass_mg = 23.985041697
    mass_h = 1.00782503223
    mu_cah = mass_ca * mass_h / (mass_ca + mass_h)
    mu_mgh = mass_mg * mass_h / (mass_mg + mass_h)
    mgh_energy_scale = mu_cah / mu_mgh
    mgh = deform_related_branch_model(
        cah,
        material="MgH+",
        energy_scale=mgh_energy_scale,
        duration_scale=1.05,
        mean_pulse_retention=0.96,
        action_jitter=0.025,
        final_state_crosstalk=0.0015,
        temperature_kelvin=300.0,
        seed=1107,
        relation_note=(
            "Diatomic-hydride transfer benchmark preserving the CaH+ level/action "
            "topology while perturbing pulse efficiency and final-state leakage."
        ),
    )

    tasks: list[MoleculeTask] = []
    tasks.append(
        _task_from_model(
            task_id=0,
            name="CaH+",
            family="diatomic_hydride",
            transfer_group="metal_hydride",
            model=cah,
            atom_graph=build_cah_chemistry(),
            data_dir=data_dir,
            experimental_features=_experimental_features(
                temperature_kelvin=300.0,
                magnetic_field_mt=0.36,
                motional_frequency_mhz=5.164,
                logic_ion_mass_u=40.0,
                infidelity_threshold=0.01,
                lamb_dicke=0.09,
                model_fidelity_flag=0.7,
                configuration_scale=1.0,
            ),
        )
    )
    tasks.append(
        _task_from_model(
            task_id=1,
            name="MgH+",
            family="diatomic_hydride",
            transfer_group="metal_hydride",
            model=mgh,
            atom_graph=build_mgh_chemistry(),
            data_dir=data_dir,
            experimental_features=_experimental_features(
                temperature_kelvin=300.0,
                magnetic_field_mt=0.36,
                motional_frequency_mhz=5.164,
                logic_ion_mass_u=40.0,
                infidelity_threshold=0.01,
                lamb_dicke=0.09,
                model_fidelity_flag=0.25,
                configuration_scale=1.05,
            ),
            metadata={"held_out_transfer_target": True},
        )
    )

    if h3o is not None:
        d3o = deform_related_branch_model(
            h3o,
            material="D3O+",
            energy_scale=0.70,
            duration_scale=1.10,
            mean_pulse_retention=0.93,
            action_jitter=0.04,
            final_state_crosstalk=0.0025,
            temperature_kelvin=20.0,
            seed=2209,
            relation_note=(
                "Hydronium-isotopologue transfer benchmark with the H3O+ graph "
                "topology and perturbed energy/pulse maps; not a D3O+ spectrum."
            ),
        )
        tasks.append(
            _task_from_model(
                task_id=2,
                name="H3O+",
                family="hydronium_isotopologue",
                transfer_group="hydronium",
                model=h3o,
                atom_graph=build_h3o_chemistry(deuterated=False),
                data_dir=data_dir,
                experimental_features=_experimental_features(
                    temperature_kelvin=20.0,
                    magnetic_field_mt=0.36,
                    motional_frequency_mhz=5.164,
                    logic_ion_mass_u=40.0,
                    infidelity_threshold=0.01,
                    lamb_dicke=0.09,
                    model_fidelity_flag=0.5,
                    configuration_scale=1.0,
                ),
            )
        )
        tasks.append(
            _task_from_model(
                task_id=3,
                name="D3O+",
                family="hydronium_isotopologue",
                transfer_group="hydronium",
                model=d3o,
                atom_graph=build_h3o_chemistry(deuterated=True),
                data_dir=data_dir,
                experimental_features=_experimental_features(
                    temperature_kelvin=20.0,
                    magnetic_field_mt=0.36,
                    motional_frequency_mhz=5.164,
                    logic_ion_mass_u=40.0,
                    infidelity_threshold=0.01,
                    lamb_dicke=0.09,
                    model_fidelity_flag=0.2,
                    configuration_scale=1.10,
                ),
                metadata={"held_out_transfer_target": True},
            )
        )

    return MoleculeTaskRegistry(tasks)


__all__ = [
    "EXPERIMENTAL_FEATURE_DIM",
    "deform_related_branch_model",
    "build_default_registry",
]
