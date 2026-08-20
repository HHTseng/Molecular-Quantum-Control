"""Expected-branch qMDP target for heterogeneous molecule minibatches."""
from __future__ import annotations

import numpy as np
import torch

from rlqls.gnn.q_network import ChemistryConditionedQNetwork

from .env import MultiMoleculeRLQLSEnv
from .observation import observation_from_numpy, obs_stack, obs_to
from .replay import MultiReplayItem


def _stack_numpy_observations(
    observations: list[dict[str, np.ndarray]],
    device: torch.device,
):
    return obs_to(
        obs_stack([observation_from_numpy(observation) for observation in observations]),
        device,
    )


def expected_branch_target(
    env: MultiMoleculeRLQLSEnv,
    items: list[MultiReplayItem],
    online: ChemistryConditionedQNetwork,
    target: ChemistryConditionedQNetwork,
    *,
    gamma: float,
    device: torch.device,
    bootstrap_on_truncation: bool,
) -> torch.Tensor:
    r"""Multi-molecule Double-DQN qMDP target.

    For replay item ``b`` belonging to molecule ``m_b``:

    .. math::
      y_b = \sum_k p_{b,k}\left[r_{b,k}+\gamma(1-d_{b,k})
      Q_{\bar\Theta}(o'_{b,k},\arg\max_{a'\in A_{m_b}}
      Q_\Theta(o'_{b,k},a'))\right].
    """

    branch_observations: list[dict[str, np.ndarray]] = []
    probabilities: list[list[float]] = []
    rewards: list[list[float]] = []
    done: list[list[float]] = []

    for item in items:
        task = env.registry.get(item.task_name)
        details = env.branch_details(
            item.state,
            item.action,
            task=task,
            next_step_count=item.next_step_count,
        )
        probabilities.append(details.probabilities.tolist())
        rewards.append(details.rewards.tolist())
        branch_done: list[float] = []
        for k in range(2):
            branch_observations.append(
                env.builder.build(
                    task,
                    details.next_states[k],
                    step_count=item.next_step_count,
                    max_steps=env.max_steps_for(task),
                )
            )
            terminal = bool(details.terminated[k])
            if bootstrap_on_truncation:
                branch_done.append(float(terminal))
            else:
                branch_done.append(
                    float(terminal or item.next_step_count >= env.max_steps_for(task))
                )
        done.append(branch_done)

    branch_batch = _stack_numpy_observations(branch_observations, device)
    with torch.no_grad():
        online_q = online(branch_batch)
        chosen = online_q.argmax(dim=1, keepdim=True)
        target_q = target(branch_batch).gather(1, chosen).squeeze(1)
        values = target_q.reshape(len(items), 2)
        p = torch.as_tensor(probabilities, dtype=torch.float32, device=device)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        d = torch.as_tensor(done, dtype=torch.float32, device=device)
        return torch.sum(p * (r + gamma * (1.0 - d) * values), dim=1)


def sampled_double_dqn_target(
    env: MultiMoleculeRLQLSEnv,
    items: list[MultiReplayItem],
    online: ChemistryConditionedQNetwork,
    target: ChemistryConditionedQNetwork,
    *,
    gamma: float,
    device: torch.device,
    bootstrap_on_truncation: bool,
) -> torch.Tensor:
    next_observations = []
    done = []
    rewards = []
    for item in items:
        task = env.registry.get(item.task_name)
        next_observations.append(
            env.builder.build(
                task,
                item.next_state,
                step_count=item.next_step_count,
                max_steps=env.max_steps_for(task),
            )
        )
        rewards.append(item.reward)
        done.append(
            item.terminated
            if bootstrap_on_truncation
            else (item.terminated or item.truncated)
        )
    batch = _stack_numpy_observations(next_observations, device)
    with torch.no_grad():
        chosen = online(batch).argmax(dim=1, keepdim=True)
        values = target(batch).gather(1, chosen).squeeze(1)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32, device=device)
        return reward_tensor + gamma * (1.0 - done_tensor) * values


__all__ = ["expected_branch_target", "sampled_double_dqn_target"]
