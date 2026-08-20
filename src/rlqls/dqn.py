"""PyTorch DQN and expected-branch qMDP training for RL-QLS.

The neural network receives the molecular posterior population s and returns
one Q value for every pulse action:

    Q_theta(s) = (Q_theta(s,0), ..., Q_theta(s,A-1)).

During interaction, an epsilon-greedy policy chooses either a random pulse or

    a_t = argmax_a Q_theta(s_t,a).

Two Bellman targets are implemented.

Sampled Double DQN
------------------
Only the measurement branch realized by ``env.step`` is used:

    y_sample = r + gamma (1-d)
               Q_target(s', argmax_a Q_online(s',a)).

Expected-branch qMDP Double DQN
--------------------------------
The known pulse/measurement model is used to average over every motional
outcome k before sampling noise enters the target:

    y_qMDP = sum_k p(k|s,a) [
                 r_k + gamma (1-d_k)
                 Q_target(F_{a,k}(s),
                          argmax_a' Q_online(F_{a,k}(s),a'))
             ].

This is the code-level implementation of Supplemental Eq. (S18), augmented by
Double-DQN action selection/evaluation.  The paper states that both qMDP and
double-Q networks are used but does not print their combined formula; this
combination is therefore a documented implementation choice.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import random
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .env import RLQLSEnv

UpdateMode = Literal["qmdp", "sampled"]


@dataclass(slots=True)
class DQNConfig:
    """Hyperparameters for the paper-style DQN agent."""

    episodes: int = 1000
    hidden_sizes: tuple[int, ...] = (128, 128, 128)
    learning_rate: float = 5e-4
    gamma: float = 1.0
    target_tau: float = 1e-3
    replay_capacity: int = 100_000
    batch_size: int = 128
    warmup_transitions: int = 256
    train_every_steps: int = 1
    gradient_steps: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.005
    epsilon_decay_fraction: float = 0.3
    update_mode: UpdateMode = "qmdp"
    loss: Literal["smooth_l1", "mse"] = "smooth_l1"
    gradient_clip_norm: float | None = 10.0
    seed: int = 0
    device: str = "cpu"
    torch_num_threads: int | None = 1

    # The original reproduction treated the artificial max-step cutoff as an
    # absorbing failed episode, so the default below preserves those results.
    # Set True to follow the common continuing-task convention of bootstrapping
    # across a Gymnasium truncation.  Physical task completion (terminated)
    # never bootstraps.
    bootstrap_on_truncation: bool = False

    def epsilon(self, episode: int) -> float:
        """Exponentially decay exploration from epsilon_start to epsilon_end.

        The printed sign in Supplemental Eq. (S16) does not yield
        epsilon(0)=epsilon_start.  This corrected conventional expression does.
        """

        tau = max(self.epsilon_decay_fraction * self.episodes, 1e-12)
        return float(
            self.epsilon_end
            + (self.epsilon_start - self.epsilon_end) * math.exp(-episode / tau)
        )


@dataclass(frozen=True, slots=True)
class ReplayItem:
    """One sampled environment interaction stored for off-policy learning."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    next_step_count: int


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.data: deque[ReplayItem] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def append(self, item: ReplayItem) -> None:
        self.data.append(item)

    def sample(self, n: int) -> list[ReplayItem]:
        return self.rng.sample(self.data, n)

    def __len__(self) -> int:
        return len(self.data)


class QNetwork(nn.Module):
    """Fully connected approximation to the pulse action-value function.

    Input dimension:
        N molecular population components.
    Output dimension:
        A pulse values, one per discrete action.

    The default ``(128,128,128)`` architecture follows the paper's three hidden
    layers with 128 nodes per layer.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_states
        for hidden_width in hidden_sizes:
            layers.extend([nn.Linear(width, hidden_width), nn.ReLU()])
            width = hidden_width
        layers.append(nn.Linear(width, n_actions))
        self.net = nn.Sequential(*layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Return shape ``(batch,A)`` from input shape ``(batch,N)``."""

        return self.net(state_batch)


@dataclass(slots=True)
class TrainingHistory:
    episode_lengths: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_success: list[bool] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    total_environment_steps: int = 0
    optimizer_steps: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class TrainedDQN:
    online: QNetwork
    target: QNetwork
    config: DQNConfig
    history: TrainingHistory

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "config": asdict(self.config),
                "history": self.history.as_dict(),
            },
            path,
        )


def _soft_update(target: QNetwork, online: QNetwork, tau: float) -> None:
    """Polyak update: theta_target <- (1-tau) theta_target + tau theta_online."""

    with torch.no_grad():
        for target_parameter, online_parameter in zip(
            target.parameters(),
            online.parameters(),
        ):
            target_parameter.mul_(1.0 - tau).add_(online_parameter, alpha=tau)


def greedy_action(
    network: QNetwork,
    state: np.ndarray,
    device: torch.device,
) -> int:
    """Select the pulse index with maximum predicted Q value."""

    with torch.no_grad():
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        return int(network(state_tensor).argmax(dim=1).item())


def _qmdp_target_torch(
    env: RLQLSEnv,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_step_counts: torch.Tensor,
    online: QNetwork,
    target: QNetwork,
    gamma: float,
    branch_maps: torch.Tensor,
    bbr_maps: torch.Tensor | None,
    *,
    bootstrap_on_truncation: bool,
) -> torch.Tensor:
    """Vectorized expected-branch Double-DQN target.

    This function intentionally recomputes *both* measurement branches from the
    replayed ``(s,a)`` pair.  The sampled ``next_state`` stored in replay is not
    used in qMDP mode because Supplemental Eq. (S18) averages over the exact
    branch probabilities instead of one Monte Carlo outcome.
    """

    # Select the two B[a,k] matrices for every replay item.
    maps = branch_maps[actions]  # (batch,2,N,N)

    # q_{a,k}=B[a,k]s.  Broadcasting inserts the outcome axis.
    raw = torch.matmul(maps, states[:, None, :, None]).squeeze(-1)  # (B,2,N)
    raw_mass = raw.sum(dim=2)  # (B,2)

    # p(k|s,a).  The denominator should already equal one for trace-preserving
    # maps, but explicit normalization protects against roundoff.
    probabilities = raw_mass / raw_mass.sum(dim=1, keepdim=True).clamp_min(1e-15)

    # F_{a,k}(s)=q_{a,k}/p(k|s,a).
    safe_mass = raw_mass.clamp_min(1e-15)
    next_states = raw / safe_mass[:, :, None]
    zero_probability = raw_mass <= 1e-15
    if torch.any(zero_probability):
        next_states = torch.where(
            zero_probability[:, :, None],
            states[:, None, :],
            next_states,
        )

    # Optional blackbody-radiation population propagation after conditioning.
    if bbr_maps is not None and env.apply_bbr:
        noise = bbr_maps[actions]  # (B,N,N)
        next_states = torch.matmul(
            noise[:, None, :, :],
            next_states[:, :, :, None],
        ).squeeze(-1)
        next_states = next_states.clamp_min(0.0)
        next_states = next_states / next_states.sum(dim=2, keepdim=True).clamp_min(1e-15)

    # Branch-dependent physics-informed reward.
    dot = torch.sum(states[:, None, :] * next_states, dim=2)
    norm = (
        torch.linalg.vector_norm(states, dim=1)[:, None]
        * torch.linalg.vector_norm(next_states, dim=2)
    )
    overlaps = torch.where(
        norm > 0.0,
        dot / norm.clamp_min(1e-15),
        torch.ones_like(dot),
    )
    rewards = torch.full_like(probabilities, env.base_reward)
    if env.overlap_penalty > 0.0:
        rewards = rewards - env.overlap_penalty * (
            overlaps > env.overlap_threshold
        ).to(rewards.dtype)

    # Physical success is a true MDP terminal state.
    terminal = torch.amax(next_states, dim=2) >= env.confidence_threshold

    # ``max_steps`` is a simulation cutoff, not a physical state property.
    # The default configuration treats it as absorbing failure to preserve the
    # reproduction runs.  Set bootstrap_on_truncation=True to bootstrap instead.
    if bootstrap_on_truncation:
        done = terminal
    else:
        time_limit = next_step_counts[:, None] >= env.max_steps
        done = torch.logical_or(terminal, time_limit)
    done_float = done.to(torch.float32)

    flat_next = next_states.reshape(-1, env.model.n_states)
    with torch.no_grad():
        # Double DQN: online network selects, target network evaluates.
        chosen_actions = online(flat_next).argmax(dim=1, keepdim=True)
        branch_values = target(flat_next).gather(1, chosen_actions)
        branch_values = branch_values.reshape(states.shape[0], 2)

        branch_targets = rewards + gamma * (1.0 - done_float) * branch_values
        return torch.sum(probabilities * branch_targets, dim=1)


def _sampled_target(
    items: list[ReplayItem],
    online: QNetwork,
    target: QNetwork,
    gamma: float,
    device: torch.device,
    *,
    bootstrap_on_truncation: bool,
) -> torch.Tensor:
    """Ordinary sampled Double-DQN target from realized transitions."""

    next_states = torch.as_tensor(
        np.stack([item.next_state for item in items]),
        dtype=torch.float32,
        device=device,
    )
    rewards = torch.as_tensor(
        [item.reward for item in items],
        dtype=torch.float32,
        device=device,
    )

    if bootstrap_on_truncation:
        done_values = [item.terminated for item in items]
    else:
        done_values = [item.terminated or item.truncated for item in items]
    done = torch.as_tensor(done_values, dtype=torch.float32, device=device)

    with torch.no_grad():
        chosen_actions = online(next_states).argmax(dim=1, keepdim=True)
        values = target(next_states).gather(1, chosen_actions).squeeze(1)
        return rewards + gamma * (1.0 - done) * values


def train_dqn(
    env: RLQLSEnv,
    config: DQNConfig,
    *,
    initial_online_state: dict[str, torch.Tensor] | None = None,
    initial_target_state: dict[str, torch.Tensor] | None = None,
) -> TrainedDQN:
    """Interact with the Gym environment and optimize the pulse-value network.

    Data-flow summary
    -----------------
    1. ``state = env.reset()`` gives a thermal molecular population.
    2. The epsilon-greedy policy chooses a pulse index.
    3. ``env.step(action)`` samples one motional measurement and returns the
       conditioned molecular population.
    4. The realized transition is stored in replay.
    5. A replay minibatch updates Q using either the sampled or qMDP target.
    6. The target network is softly moved toward the online network.
    """

    if config.torch_num_threads is not None:
        torch.set_num_threads(config.torch_num_threads)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    online = QNetwork(
        env.model.n_states,
        env.model.n_actions,
        config.hidden_sizes,
    ).to(device)
    target = QNetwork(
        env.model.n_states,
        env.model.n_actions,
        config.hidden_sizes,
    ).to(device)

    if initial_online_state is not None:
        online.load_state_dict(initial_online_state)
    if initial_target_state is not None:
        target.load_state_dict(initial_target_state)
    else:
        target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(
        online.parameters(),
        lr=config.learning_rate,
    )
    replay = ReplayBuffer(config.replay_capacity, config.seed)
    history = TrainingHistory()
    global_step = 0

    # Copy the physics transition kernel to the training device once.  qMDP
    # then computes both branches with batched tensor operations.
    branch_maps = torch.as_tensor(
        env.model.branch_matrices,
        dtype=torch.float32,
        device=device,
    )
    bbr_maps = (
        None
        if env.model.bbr_propagators is None
        else torch.as_tensor(
            env.model.bbr_propagators,
            dtype=torch.float32,
            device=device,
        )
    )

    for episode in range(config.episodes):
        state, _ = env.reset(seed=config.seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_steps = 0
        epsilon = config.epsilon(episode)

        while not (terminated or truncated):
            # Exploration changes the pulse selected, not the measurement
            # result.  The latter remains sampled by the quantum environment.
            if np.random.random() < epsilon:
                action = int(np.random.randint(env.model.n_actions))
            else:
                action = greedy_action(online, state, device)

            next_state, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            global_step += 1
            episode_return += reward

            replay.append(
                ReplayItem(
                    state=state.copy(),
                    action=action,
                    reward=reward,
                    next_state=next_state.copy(),
                    terminated=terminated,
                    truncated=truncated,
                    next_step_count=int(info["step_count"]),
                )
            )
            state = next_state

            enough_data = len(replay) >= max(
                config.batch_size,
                config.warmup_transitions,
            )
            update_due = global_step % config.train_every_steps == 0
            if enough_data and update_due:
                for _ in range(config.gradient_steps):
                    items = replay.sample(config.batch_size)
                    states = torch.as_tensor(
                        np.stack([item.state for item in items]),
                        dtype=torch.float32,
                        device=device,
                    )
                    actions = torch.as_tensor(
                        [item.action for item in items],
                        dtype=torch.long,
                        device=device,
                    )

                    # Q_theta(s,a): select the network output corresponding to
                    # the pulse that was actually taken in each replay item.
                    prediction = online(states).gather(
                        1,
                        actions[:, None],
                    ).squeeze(1)

                    if config.update_mode == "qmdp":
                        next_step_counts = torch.as_tensor(
                            [item.next_step_count for item in items],
                            dtype=torch.long,
                            device=device,
                        )
                        expected = _qmdp_target_torch(
                            env=env,
                            states=states,
                            actions=actions,
                            next_step_counts=next_step_counts,
                            online=online,
                            target=target,
                            gamma=config.gamma,
                            branch_maps=branch_maps,
                            bbr_maps=bbr_maps,
                            bootstrap_on_truncation=config.bootstrap_on_truncation,
                        )
                    else:
                        expected = _sampled_target(
                            items=items,
                            online=online,
                            target=target,
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
                            online.parameters(),
                            config.gradient_clip_norm,
                        )
                    optimizer.step()
                    _soft_update(target, online, config.target_tau)

                    history.losses.append(float(loss.detach().cpu()))
                    history.optimizer_steps += 1

        history.episode_lengths.append(episode_steps)
        history.episode_returns.append(episode_return)
        history.episode_success.append(bool(terminated))
        history.epsilons.append(epsilon)

    history.total_environment_steps = global_step
    return TrainedDQN(online, target, config, history)


__all__ = [
    "DQNConfig",
    "QNetwork",
    "TrainingHistory",
    "TrainedDQN",
    "train_dqn",
    "greedy_action",
]
