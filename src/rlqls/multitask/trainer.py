"""Shared GNN Double-DQN/qMDP training across molecule-specific MDPs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import math
from pathlib import Path
import random
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rlqls.gnn import ChemistryConditionedQNetwork, GNNQConfig

from .env import MultiMoleculeRLQLSEnv
from .observation import observation_from_numpy, obs_stack, obs_to
from .qmdp import expected_branch_target, sampled_double_dqn_target
from .replay import BalancedReplayBuffer, MultiReplayItem


UpdateMode = Literal["qmdp", "sampled"]
TaskSchedule = Literal["round_robin", "random"]


@dataclass(slots=True)
class MultiTaskDQNConfig:
    episodes: int = 400
    training_tasks: tuple[str, ...] = ("CaH+", "H3O+")
    learning_rate: float = 3e-4
    gamma: float = 0.95
    target_tau: float = 2e-3
    replay_capacity_per_task: int = 20_000
    batch_size: int = 32
    warmup_transitions: int = 64
    train_every_steps: int = 2
    gradient_steps: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.5
    update_mode: UpdateMode = "qmdp"
    task_schedule: TaskSchedule = "round_robin"
    loss: Literal["smooth_l1", "mse"] = "smooth_l1"
    gradient_clip_norm: float | None = 5.0
    bootstrap_on_truncation: bool = False
    seed: int = 0
    device: str = "cpu"
    torch_num_threads: int | None = 1
    freeze_chemistry_encoder: bool = False
    gnn: GNNQConfig = field(default_factory=GNNQConfig)

    def epsilon(self, episode: int) -> float:
        tau = max(self.epsilon_decay_fraction * self.episodes, 1e-12)
        return float(
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end) * math.exp(-episode / tau)
        )


@dataclass(slots=True)
class MultiTaskTrainingHistory:
    episode_tasks: list[str] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_success: list[bool] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    mean_losses: list[float] = field(default_factory=list)
    total_environment_steps: int = 0
    optimizer_steps: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def task_summary(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for task in sorted(set(self.episode_tasks)):
            indices = [i for i, name in enumerate(self.episode_tasks) if name == task]
            lengths = [self.episode_lengths[i] for i in indices]
            success = [self.episode_success[i] for i in indices]
            result[task] = {
                "episodes": len(indices),
                "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
                "success_rate": float(np.mean(success)) if success else float("nan"),
            }
        return result


@dataclass(slots=True)
class TrainedMultiTaskDQN:
    online: ChemistryConditionedQNetwork
    target: ChemistryConditionedQNetwork
    config: MultiTaskDQNConfig
    history: MultiTaskTrainingHistory

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "config": {
                    **asdict(self.config),
                    "gnn": self.config.gnn.as_dict(),
                },
                "history": self.history.as_dict(),
            },
            path,
        )
        return path


def build_q_network(
    env: MultiMoleculeRLQLSEnv,
    config: GNNQConfig,
) -> ChemistryConditionedQNetwork:
    d = env.registry.feature_dimensions
    return ChemistryConditionedQNetwork(
        atom_dim=d["atom"],
        atom_edge_dim=d["atom_edge"],
        explicit_chemistry_dim=d["explicit"],
        level_dim=env.builder.level_input_dim,
        level_edge_dim=d["level_edge"],
        pulse_dim=d["pulse"],
        pulse_transition_dim=d["pulse_transition"],
        config=config,
    )


def clone_network(
    env: MultiMoleculeRLQLSEnv,
    source: ChemistryConditionedQNetwork,
) -> ChemistryConditionedQNetwork:
    network = build_q_network(env, source.config)
    network.load_state_dict(copy.deepcopy(source.state_dict()))
    return network


def _single_observation_batch(observation: dict[str, np.ndarray], device: torch.device):
    return obs_to(observation_from_numpy(observation), device)


def greedy_action(
    network: ChemistryConditionedQNetwork,
    observation: dict[str, np.ndarray],
    device: torch.device | str,
) -> int:
    network.eval()
    with torch.no_grad():
        q = network(_single_observation_batch(observation, torch.device(device)))
        return int(torch.argmax(q).item())


def _soft_update(
    target: nn.Module,
    online: nn.Module,
    tau: float,
) -> None:
    with torch.no_grad():
        for target_parameter, online_parameter in zip(
            target.parameters(), online.parameters()
        ):
            target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)


def _current_observation_batch(
    env: MultiMoleculeRLQLSEnv,
    items: list[MultiReplayItem],
    device: torch.device,
):
    observations = []
    for item in items:
        task = env.registry.get(item.task_name)
        observations.append(
            env.builder.build(
                task,
                item.state,
                step_count=item.step_count,
                max_steps=env.max_steps_for(task),
            )
        )
    return obs_to(
        obs_stack([observation_from_numpy(observation) for observation in observations]),
        device,
    )


def _select_episode_task(
    config: MultiTaskDQNConfig,
    episode: int,
    rng: np.random.Generator,
) -> str:
    if config.task_schedule == "round_robin":
        return config.training_tasks[episode % len(config.training_tasks)]
    return config.training_tasks[int(rng.integers(len(config.training_tasks)))]


def train_multitask_dqn(
    env: MultiMoleculeRLQLSEnv,
    config: MultiTaskDQNConfig,
    *,
    initial_online_state: dict[str, torch.Tensor] | None = None,
    initial_target_state: dict[str, torch.Tensor] | None = None,
) -> TrainedMultiTaskDQN:
    """Train one GNN parameter set on several molecule-specific MDPs."""

    if not config.training_tasks:
        raise ValueError("training_tasks cannot be empty")
    for name in config.training_tasks:
        env.registry.get(name)
    if config.torch_num_threads is not None:
        torch.set_num_threads(config.torch_num_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)

    online = build_q_network(env, config.gnn).to(device)
    target = build_q_network(env, config.gnn).to(device)
    if initial_online_state is not None:
        online.load_state_dict(initial_online_state)
    if initial_target_state is not None:
        target.load_state_dict(initial_target_state)
    else:
        target.load_state_dict(online.state_dict())
    target.eval()

    if config.freeze_chemistry_encoder:
        for parameter in online.chemistry_encoder.parameters():
            parameter.requires_grad_(False)
    trainable = [parameter for parameter in online.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
    replay = BalancedReplayBuffer(config.replay_capacity_per_task, config.seed)
    history = MultiTaskTrainingHistory()
    global_step = 0

    for episode in range(config.episodes):
        task_name = _select_episode_task(config, episode, rng)
        observation, info = env.reset(
            seed=config.seed + 1009 * episode,
            options={"molecule": task_name},
        )
        state = np.asarray(info["raw_population"], dtype=np.float64)
        terminated = truncated = False
        episode_return = 0.0
        episode_losses: list[float] = []
        epsilon = config.epsilon(episode)

        while not (terminated or truncated):
            if rng.random() < epsilon:
                valid = np.flatnonzero(info["action_mask"])
                action = int(rng.choice(valid))
            else:
                action = greedy_action(online, observation, device)

            step_count = int(info["step_count"])
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            next_state = np.asarray(next_info["raw_population"], dtype=np.float64)
            replay.append(
                MultiReplayItem(
                    task_name=task_name,
                    state=state.copy(),
                    action=action,
                    reward=float(reward),
                    next_state=next_state.copy(),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    step_count=step_count,
                    next_step_count=int(next_info["step_count"]),
                )
            )
            state = next_state
            observation = next_observation
            info = next_info
            episode_return += float(reward)
            global_step += 1

            if (
                len(replay) >= config.warmup_transitions
                and global_step % config.train_every_steps == 0
            ):
                for _ in range(config.gradient_steps):
                    items = replay.sample(
                        config.batch_size,
                        task_names=config.training_tasks,
                    )
                    current_batch = _current_observation_batch(env, items, device)
                    actions = torch.as_tensor(
                        [item.action for item in items],
                        dtype=torch.int64,
                        device=device,
                    )
                    prediction = online(current_batch).gather(
                        1, actions.unsqueeze(1)
                    ).squeeze(1)
                    if config.update_mode == "qmdp":
                        expected = expected_branch_target(
                            env,
                            items,
                            online,
                            target,
                            gamma=config.gamma,
                            device=device,
                            bootstrap_on_truncation=config.bootstrap_on_truncation,
                        )
                    else:
                        expected = sampled_double_dqn_target(
                            env,
                            items,
                            online,
                            target,
                            gamma=config.gamma,
                            device=device,
                            bootstrap_on_truncation=config.bootstrap_on_truncation,
                        )
                    if config.loss == "smooth_l1":
                        loss = F.smooth_l1_loss(prediction, expected)
                    else:
                        loss = F.mse_loss(prediction, expected)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if config.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(
                            trainable, config.gradient_clip_norm
                        )
                    optimizer.step()
                    _soft_update(target, online, config.target_tau)
                    episode_losses.append(float(loss.item()))
                    history.optimizer_steps += 1

        history.episode_tasks.append(task_name)
        history.episode_lengths.append(int(info["step_count"]))
        history.episode_returns.append(episode_return)
        history.episode_success.append(bool(terminated))
        history.epsilons.append(epsilon)
        history.mean_losses.append(
            float(np.mean(episode_losses)) if episode_losses else float("nan")
        )

    history.total_environment_steps = global_step
    return TrainedMultiTaskDQN(online, target, config, history)


__all__ = [
    "MultiTaskDQNConfig",
    "MultiTaskTrainingHistory",
    "TrainedMultiTaskDQN",
    "build_q_network",
    "clone_network",
    "greedy_action",
    "train_multitask_dqn",
]


def load_network_checkpoint(
    env: MultiMoleculeRLQLSEnv,
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[ChemistryConditionedQNetwork, dict]:
    """Reconstruct a shared GNN and load an implementation checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config_data = checkpoint.get("config", {})
    gnn_data = config_data.get("gnn", {})
    gnn_config = GNNQConfig(**gnn_data) if gnn_data else GNNQConfig()
    network = build_q_network(env, gnn_config).to(torch.device(device))
    network.load_state_dict(checkpoint["online"])
    network.eval()
    return network, checkpoint


__all__.append("load_network_checkpoint")
