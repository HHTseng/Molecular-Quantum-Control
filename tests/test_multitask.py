from pathlib import Path

import numpy as np
import torch

from rlqls.gnn import GNNQConfig
from rlqls.multitask import MultiMoleculeRLQLSEnv, build_default_registry
from rlqls.multitask.observation import observation_from_numpy, obs_stack
from rlqls.multitask.trainer import MultiTaskDQNConfig, build_q_network, train_multitask_dqn


ROOT = Path(__file__).resolve().parents[1]


def build_registry():
    return build_default_registry(
        ROOT / "data",
        precomputed_dir=ROOT / "results",
    )


def test_registry_and_action_masks():
    registry = build_registry()
    assert registry.names == ("CaH+", "MgH+", "H3O+", "D3O+")
    assert registry.get("CaH+").n_actions == 13
    assert registry.get("H3O+").n_actions == 218
    env = MultiMoleculeRLQLSEnv(registry, max_steps=5)
    for name in registry.names:
        observation, info = env.reset(seed=3, options={"molecule": name})
        assert int(np.sum(info["action_mask"])) == registry.get(name).n_actions
        assert observation["level_features"].shape[0] == registry.max_states
        assert np.isclose(info["raw_population"].sum(), 1.0)


def test_shared_gnn_scores_variable_actions_and_masks_padding():
    registry = build_registry()
    env = MultiMoleculeRLQLSEnv(registry, max_steps=5)
    observations = []
    for name in registry.names:
        observation, _ = env.reset(seed=5, options={"molecule": name})
        observations.append(observation_from_numpy(observation))
    batch = obs_stack(observations)
    network = build_q_network(
        env,
        GNNQConfig(
            hidden_dim=16,
            chemistry_context_dim=16,
            chemistry_layers=1,
            spectroscopy_layers=1,
        ),
    )
    q_values = network(batch)
    assert q_values.shape == (4, registry.max_actions)
    for index, task in enumerate(registry.tasks):
        assert torch.all(torch.isfinite(q_values[index, : task.n_actions]))
        if task.n_actions < registry.max_actions:
            assert torch.all(q_values[index, task.n_actions :] < -1e8)


def test_multitask_qmdp_training_smoke():
    registry = build_registry()
    env = MultiMoleculeRLQLSEnv(
        registry,
        allowed_tasks=("CaH+", "MgH+"),
        max_steps={"CaH+": 6, "MgH+": 6},
    )
    trained = train_multitask_dqn(
        env,
        MultiTaskDQNConfig(
            episodes=4,
            training_tasks=("CaH+", "MgH+"),
            batch_size=4,
            warmup_transitions=4,
            train_every_steps=2,
            seed=11,
            gnn=GNNQConfig(
                hidden_dim=16,
                chemistry_context_dim=16,
                chemistry_layers=1,
                spectroscopy_layers=1,
            ),
        ),
    )
    assert trained.history.optimizer_steps > 0
    assert set(trained.history.episode_tasks) == {"CaH+", "MgH+"}


def test_related_surrogates_remain_terminal_reachable_at_099():
    registry = build_registry()
    for name in ("MgH+", "D3O+"):
        # This is a necessary, not sufficient, reachability diagnostic.  It
        # guards against adding a per-step leakage floor larger than the
        # environment's 1% infidelity tolerance.
        assert registry.get(name).maximum_one_step_conditional_purity() >= 0.99
