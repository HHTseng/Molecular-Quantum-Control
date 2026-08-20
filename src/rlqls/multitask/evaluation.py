"""Independent evaluation utilities for shared-GNN transfer experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch

from rlqls.gnn.q_network import ChemistryConditionedQNetwork

from .env import MultiMoleculeRLQLSEnv
from .observation import observation_from_numpy, obs_stack, obs_to


@dataclass(slots=True)
class TaskEvaluationResult:
    task_name: str
    policy_name: str
    lengths: list[int]
    returns: list[float]
    successes: list[bool]
    terminal_states: list[int]

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes))

    @property
    def mean_censored_length(self) -> float:
        return float(np.mean(self.lengths))

    @property
    def mean_successful_length(self) -> float:
        values = [length for length, success in zip(self.lengths, self.successes) if success]
        return float(np.mean(values)) if values else float("nan")

    def completion_fraction(self, horizon: int) -> float:
        return float(
            np.mean(
                [
                    success and length <= horizon
                    for length, success in zip(self.lengths, self.successes)
                ]
            )
        )

    def summary(self, horizons: tuple[int, ...] = ()) -> dict:
        output = {
            "task": self.task_name,
            "policy": self.policy_name,
            "episodes": len(self.lengths),
            "success_rate": self.success_rate,
            "mean_censored_length": self.mean_censored_length,
            "mean_successful_length": self.mean_successful_length,
            "median_censored_length": float(np.median(self.lengths)),
        }
        if horizons:
            output["completion_fraction"] = {
                str(horizon): self.completion_fraction(horizon) for horizon in horizons
            }
        return output

    def as_dict(self) -> dict:
        return asdict(self)


def _batched_observation(
    env: MultiMoleculeRLQLSEnv,
    task_name: str,
    states: np.ndarray,
    step: int,
    device: torch.device,
):
    task = env.registry.get(task_name)
    observations = [
        env.builder.build(
            task,
            state,
            step_count=step,
            max_steps=env.max_steps_for(task),
        )
        for state in states
    ]
    return obs_to(
        obs_stack([observation_from_numpy(observation) for observation in observations]),
        device,
    )


def evaluate_gnn_batched(
    env: MultiMoleculeRLQLSEnv,
    network: ChemistryConditionedQNetwork,
    *,
    task_name: str,
    episodes: int,
    seed: int,
    device: str = "cpu",
    policy_name: str = "GNN",
    inference_batch_size: int = 32,
    torch_num_threads: int | None = 1,
) -> TaskEvaluationResult:
    if torch_num_threads is not None:
        torch.set_num_threads(int(torch_num_threads))
    task = env.registry.get(task_name)
    max_steps = env.max_steps_for(task)
    rng = np.random.default_rng(seed)
    states = np.repeat(task.model.initial_population[None, :], episodes, axis=0).astype(
        np.float64
    )
    active = np.ones(episodes, dtype=bool)
    lengths = np.full(episodes, max_steps, dtype=np.int64)
    returns = np.zeros(episodes, dtype=np.float64)
    successes = np.zeros(episodes, dtype=bool)
    terminal_states = np.argmax(states, axis=1).astype(np.int64)
    torch_device = torch.device(device)
    network.eval()

    for step in range(max_steps):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        current = states[indices]
        # Large molecules have up to O(10^2) level nodes and O(10^2) pulse
        # candidates per observation.  Evaluating hundreds of Monte Carlo
        # episodes in one giant GNN batch can cause avoidable memory pressure.
        # Chunking changes neither the policy nor the stochastic trajectories.
        action_chunks: list[np.ndarray] = []
        for start in range(0, indices.size, max(int(inference_batch_size), 1)):
            stop = min(start + max(int(inference_batch_size), 1), indices.size)
            batch = _batched_observation(
                env, task_name, current[start:stop], step, torch_device
            )
            with torch.no_grad():
                action_chunks.append(
                    network(batch).argmax(dim=1).cpu().numpy().astype(np.int64)
                )
        actions = np.concatenate(action_chunks)
        probabilities, branches = task.model.batch_branches(
            current, actions, apply_bbr=env.apply_bbr
        )
        outcomes = (rng.random(indices.size) >= probabilities[:, 0]).astype(np.int64)
        selected = branches[np.arange(indices.size), outcomes]
        dot = np.sum(current * selected, axis=1)
        norm = np.linalg.norm(current, axis=1) * np.linalg.norm(selected, axis=1)
        overlap = np.divide(dot, norm, out=np.ones_like(dot), where=norm > 0.0)
        reward = np.full(indices.size, env.base_reward, dtype=np.float64)
        penalty = env._task_overlap_penalty(task)
        if penalty > 0.0:
            threshold = (
                env.overlap_threshold_override
                if env.overlap_threshold_override is not None
                else 1.0 - 1.0 / task.n_states
            )
            reward -= penalty * (overlap > threshold)
        returns[indices] += reward
        states[indices] = selected
        terminal_states[indices] = np.argmax(selected, axis=1)
        finished = np.max(selected, axis=1) >= env.confidence_threshold
        if np.any(finished):
            done_indices = indices[finished]
            successes[done_indices] = True
            lengths[done_indices] = step + 1
            active[done_indices] = False

    return TaskEvaluationResult(
        task_name,
        policy_name,
        lengths.tolist(),
        returns.tolist(),
        successes.tolist(),
        terminal_states.tolist(),
    )


def _evaluate_action_rule(
    env: MultiMoleculeRLQLSEnv,
    *,
    task_name: str,
    episodes: int,
    seed: int,
    policy_name: str,
    action_rule: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
) -> TaskEvaluationResult:
    task = env.registry.get(task_name)
    max_steps = env.max_steps_for(task)
    rng = np.random.default_rng(seed)
    states = np.repeat(task.model.initial_population[None, :], episodes, axis=0).astype(
        np.float64
    )
    active = np.ones(episodes, dtype=bool)
    lengths = np.full(episodes, max_steps, dtype=np.int64)
    returns = np.zeros(episodes, dtype=np.float64)
    successes = np.zeros(episodes, dtype=bool)
    terminal_states = np.argmax(states, axis=1).astype(np.int64)

    for step in range(max_steps):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        current = states[indices]
        actions = action_rule(current, step, rng).astype(np.int64)
        probabilities, branches = task.model.batch_branches(
            current, actions, apply_bbr=env.apply_bbr
        )
        outcomes = (rng.random(indices.size) >= probabilities[:, 0]).astype(np.int64)
        selected = branches[np.arange(indices.size), outcomes]
        returns[indices] += env.base_reward
        states[indices] = selected
        terminal_states[indices] = np.argmax(selected, axis=1)
        finished = np.max(selected, axis=1) >= env.confidence_threshold
        if np.any(finished):
            done_indices = indices[finished]
            successes[done_indices] = True
            lengths[done_indices] = step + 1
            active[done_indices] = False

    return TaskEvaluationResult(
        task_name,
        policy_name,
        lengths.tolist(),
        returns.tolist(),
        successes.tolist(),
        terminal_states.tolist(),
    )


def evaluate_random(
    env: MultiMoleculeRLQLSEnv,
    *,
    task_name: str,
    episodes: int,
    seed: int,
) -> TaskEvaluationResult:
    task = env.registry.get(task_name)
    return _evaluate_action_rule(
        env,
        task_name=task_name,
        episodes=episodes,
        seed=seed,
        policy_name="random",
        action_rule=lambda states, _step, rng: rng.integers(
            0, task.n_actions, size=states.shape[0]
        ),
    )


def evaluate_sweeping(
    env: MultiMoleculeRLQLSEnv,
    *,
    task_name: str,
    episodes: int,
    seed: int,
) -> TaskEvaluationResult:
    task = env.registry.get(task_name)
    order = list(task.model.metadata.get("sweeping_order", range(task.n_actions)))
    return _evaluate_action_rule(
        env,
        task_name=task_name,
        episodes=episodes,
        seed=seed,
        policy_name="sweeping",
        action_rule=lambda states, step, _rng: np.full(
            states.shape[0], order[step % len(order)], dtype=np.int64
        ),
    )


def chemistry_embeddings(
    env: MultiMoleculeRLQLSEnv,
    network: ChemistryConditionedQNetwork,
    *,
    task_names: tuple[str, ...] | None = None,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    names = task_names or env.registry.names
    observations = []
    for name in names:
        task = env.registry.get(name)
        observations.append(
            env.builder.build(
                task,
                task.model.initial_population,
                step_count=0,
                max_steps=env.max_steps_for(task),
            )
        )
    batch = obs_to(
        obs_stack([observation_from_numpy(observation) for observation in observations]),
        torch.device(device),
    )
    network.eval()
    with torch.no_grad():
        _, auxiliary = network(batch, return_aux=True)
    embedding = auxiliary["chemistry_embedding"].cpu().numpy()
    return {name: embedding[index] for index, name in enumerate(names)}


__all__ = [
    "TaskEvaluationResult",
    "evaluate_gnn_batched",
    "evaluate_random",
    "evaluate_sweeping",
    "chemistry_embeddings",
]
