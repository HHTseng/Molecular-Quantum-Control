"""PyTorch DQN and expected-branch qMDP training for RL-QLS.

The online network approximates ``Q_theta(s,a)``.  The qMDP update integrates
over every quantum-measurement branch, whereas sampled DDQN uses only the
realized transition.  Equations follow pseudocode Secs. 7--14.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import math, random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .env import RLQLSEnv

UpdateMode = Literal["qmdp", "sampled"]


@dataclass(slots=True)
class DQNConfig:
    """Hyperparameters in the notation of pseudocode Secs. 9--13."""
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

    def epsilon(self, episode: int) -> float:
        """Compute ``eps(n)=eps_f+(eps_0-eps_f) exp(-n/tau_eps)``."""
        # Corrected sign relative to printed Supplemental Eq. (S16), so eps(0)=eps_start.
        tau = max(self.epsilon_decay_fraction * self.episodes, 1e-12)
        return float(
            self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-episode / tau)
        )


@dataclass(frozen=True, slots=True)
class ReplayItem:
    """One sampled tuple ``(s,a,r,s',terminated,truncated)`` (Sec. 8)."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    next_step_count: int


class ReplayBuffer:
    """Uniform finite replay memory used to decorrelate trajectory samples."""
    def __init__(self, capacity: int, seed: int):
        self.data: deque[ReplayItem] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def append(self, x: ReplayItem):
        self.data.append(x)

    def sample(self, n: int):
        return self.rng.sample(self.data, n)

    def __len__(self):
        return len(self.data)


class QNetwork(nn.Module):
    """MLP mapping ``s in Delta_(N_S-1)`` to ``(Q(s,a))_(a=0)^(N_A-1)``."""
    def __init__(self, n_states: int, n_actions: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers = []
        width = n_states
        for h in hidden_sizes:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, n_actions))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Evaluate all pulse-action values for one state or a state batch."""
        return self.net(x)


@dataclass(slots=True)
class TrainingHistory:
    """Learning curves and counters; returns equal ``sum_t r_t`` per episode."""
    episode_lengths: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_success: list[bool] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    total_environment_steps: int = 0
    optimizer_steps: int = 0

    def as_dict(self):
        return asdict(self)


@dataclass(slots=True)
class TrainedDQN:
    """Online parameters ``theta``, target parameters ``theta_bar``, and history."""
    online: QNetwork
    target: QNetwork
    config: DQNConfig
    history: TrainingHistory

    def save(self, path: str | Path):
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


def _soft_update(target, online, tau):
    """Apply ``theta_bar <- (1-tau) theta_bar + tau theta`` (Sec. 12)."""
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.mul_(1 - tau).add_(op, alpha=tau)


def greedy_action(network: QNetwork, state: np.ndarray, device: torch.device) -> int:
    """Return ``argmax_a Q_theta(s,a)``."""
    with torch.no_grad():
        x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(network(x).argmax(1).item())


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
) -> torch.Tensor:
    """Compute the expected-branch Double-DQN target (pseudocode Sec. 11).

    For each replay pair ``(s_b,a_b)``, this returns
    ``y_b=sum_k p_k [r_k + gamma(1-d_k) Q_target(s'_k,argmax Q_online)]``.
    """
    maps = branch_maps[actions]  # (B,2,N,N)
    raw = torch.matmul(maps, states[:, None, :, None]).squeeze(-1)  # B_{a,k}s
    raw_mass = raw.sum(dim=2)  # Born masses p_k=1^T B_{a,k}s
    probabilities = raw_mass / raw_mass.sum(dim=1, keepdim=True).clamp_min(1e-15)
    safe_mass = raw_mass.clamp_min(1e-15)
    next_states = raw / safe_mass[:, :, None]
    zero_mask = raw_mass <= 1e-15
    if torch.any(zero_mask):
        next_states = torch.where(zero_mask[:, :, None], states[:, None, :], next_states)

    if bbr_maps is not None and env.apply_bbr:
        noise = bbr_maps[actions]  # (B,N,N)
        next_states = torch.matmul(noise[:, None, :, :], next_states[:, :, :, None]).squeeze(-1)
        next_states = next_states.clamp_min(0.0)
        next_states = next_states / next_states.sum(dim=2, keepdim=True).clamp_min(1e-15)

    # Branch reward uses cosine overlap o(s,s'_k), pseudocode Sec. 2.7.
    dot = torch.sum(states[:, None, :] * next_states, dim=2)
    norm = torch.linalg.vector_norm(states, dim=1)[:, None] * torch.linalg.vector_norm(
        next_states, dim=2
    )
    overlaps = torch.where(norm > 0.0, dot / norm.clamp_min(1e-15), torch.ones_like(dot))
    rewards = torch.full_like(probabilities, env.base_reward)
    if env.overlap_penalty > 0.0:
        rewards = rewards - env.overlap_penalty * (overlaps > env.overlap_threshold).to(
            rewards.dtype
        )

    terminal = torch.amax(next_states, dim=2) >= env.confidence_threshold  # max_i s'_i >= 1-eta
    time_limit = next_step_counts[:, None] >= env.max_steps
    done = torch.logical_or(terminal, time_limit).to(torch.float32)

    flat_next = next_states.reshape(-1, env.model.n_states)
    with torch.no_grad():
        # Double DQN: theta selects a*, theta_bar evaluates Q(s',a*).
        chosen = online(flat_next).argmax(dim=1, keepdim=True)
        values = target(flat_next).gather(1, chosen).reshape(states.shape[0], 2)
        return torch.sum(probabilities * (rewards + gamma * (1.0 - done) * values), dim=1)


def _sampled_target(items, online, target, gamma, device):
    """Compute ``y=r+gamma(1-d)Q_target(s',argmax Q_online)`` (Sec. 10)."""
    ns = torch.as_tensor(
        np.stack([x.next_state for x in items]), dtype=torch.float32, device=device
    )
    r = torch.as_tensor([x.reward for x in items], dtype=torch.float32, device=device)
    d = torch.as_tensor(
        [x.terminated or x.truncated for x in items], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        chosen = online(ns).argmax(1, keepdim=True)
        values = target(ns).gather(1, chosen).squeeze(1)
        return r + gamma * (1 - d) * values


def train_dqn(
    env: RLQLSEnv,
    config: DQNConfig,
    *,
    initial_online_state: dict[str, torch.Tensor] | None = None,
    initial_target_state: dict[str, torch.Tensor] | None = None,
) -> TrainedDQN:
    """Train along sampled trajectories with qMDP or sampled DDQN targets.

    The realized branch advances ``s_t -> s_(t+1)`` in both modes.  In qMDP
    mode the optimizer target re-computes and averages both possible outcomes.
    """
    if config.torch_num_threads is not None:
        torch.set_num_threads(config.torch_num_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    online = QNetwork(env.model.n_states, env.model.n_actions, config.hidden_sizes).to(device)
    target = QNetwork(env.model.n_states, env.model.n_actions, config.hidden_sizes).to(device)
    if initial_online_state is not None:
        online.load_state_dict(initial_online_state)
    if initial_target_state is not None:
        target.load_state_dict(initial_target_state)
    else:
        target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity, config.seed)
    history = TrainingHistory()
    global_step = 0
    branch_maps = torch.as_tensor(env.model.branch_matrices, dtype=torch.float32, device=device)
    bbr_maps = (
        None
        if env.model.bbr_propagators is None
        else torch.as_tensor(env.model.bbr_propagators, dtype=torch.float32, device=device)
    )
    for episode in range(config.episodes):
        state, _ = env.reset(seed=config.seed + episode)
        terminated = truncated = False
        ret = 0.0
        steps = 0
        epsilon = config.epsilon(episode)
        while not (terminated or truncated):
            # Epsilon-greedy pulse selection: random exploration vs argmax_a Q_theta.
            action = (
                int(np.random.randint(env.model.n_actions))
                if np.random.random() < epsilon
                else greedy_action(online, state, device)
            )
            next_state, reward, terminated, truncated, info = env.step(action)
            steps += 1
            global_step += 1
            ret += reward
            replay.append(
                ReplayItem(
                    state.copy(),
                    action,
                    reward,
                    next_state.copy(),
                    terminated,
                    truncated,
                    int(info["step_count"]),
                )
            )
            state = next_state
            if (
                len(replay) >= max(config.batch_size, config.warmup_transitions)
                and global_step % config.train_every_steps == 0
            ):
                for _ in range(config.gradient_steps):
                    items = replay.sample(config.batch_size)
                    s = torch.as_tensor(
                        np.stack([x.state for x in items]), dtype=torch.float32, device=device
                    )
                    a = torch.as_tensor([x.action for x in items], dtype=torch.long, device=device)
                    prediction = online(s).gather(1, a[:, None]).squeeze(1)  # Q_theta(s_b,a_b)
                    if config.update_mode == "qmdp":
                        next_step_counts = torch.as_tensor(
                            [x.next_step_count for x in items], dtype=torch.long, device=device
                        )
                        expected = _qmdp_target_torch(
                            env,
                            s,
                            a,
                            next_step_counts,
                            online,
                            target,
                            config.gamma,
                            branch_maps,
                            bbr_maps,
                        )
                    else:
                        expected = _sampled_target(items, online, target, config.gamma, device)
                    loss = (
                        F.smooth_l1_loss(prediction, expected)
                        if config.loss == "smooth_l1"
                        else F.mse_loss(prediction, expected)
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if config.gradient_clip_norm is not None:
                        nn.utils.clip_grad_norm_(online.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    _soft_update(target, online, config.target_tau)
                    history.losses.append(float(loss.detach().cpu()))
                    history.optimizer_steps += 1
        history.episode_lengths.append(steps)
        history.episode_returns.append(ret)
        history.episode_success.append(bool(terminated))
        history.epsilons.append(epsilon)
    history.total_environment_steps = global_step
    return TrainedDQN(online, target, config, history)


__all__ = ["DQNConfig", "QNetwork", "TrainingHistory", "TrainedDQN", "train_dqn", "greedy_action"]
