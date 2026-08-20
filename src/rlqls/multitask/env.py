"""Integrated Gymnasium environment for a registry of molecular RL-QLS tasks.

One episode concerns one molecule/configuration ``m``.  The environment may
select ``m`` at reset or receive it explicitly through
``reset(options={"molecule": ...})``.  All tasks share the padded Gym action
capacity, but ``action_mask`` restricts decisions to the local pulse library
``A_m``.

For the selected molecule, the physical transition is unchanged:

    q_k = B^(m)[a,k] s,
    p(k|s,a) = 1^T q_k,
    s'_k = q_k / p(k|s,a),
    k ~ Categorical(p_0,p_1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rlqls.compat import gym, spaces

from .observation import GLOBAL_FEATURE_DIM, MolecularGraphObservationBuilder
from .registry import MoleculeTaskRegistry
from .task import MoleculeTask


@dataclass(frozen=True, slots=True)
class MultiBranchDetails:
    probabilities: np.ndarray
    next_states: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    overlaps: np.ndarray


class MultiMoleculeRLQLSEnv(gym.Env):
    """Sampled pulse-measurement environment with extendable molecule list."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        registry: MoleculeTaskRegistry,
        *,
        observation_builder: MolecularGraphObservationBuilder | None = None,
        allowed_tasks: tuple[str, ...] | None = None,
        task_probabilities: np.ndarray | None = None,
        infidelity_threshold: float = 0.01,
        max_steps: int | Mapping[str, int] = 150,
        base_reward: float = -1.0,
        overlap_penalty: float | Mapping[str, float] = 0.0,
        overlap_threshold: float | None = None,
        apply_bbr: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < infidelity_threshold < 1.0:
            raise ValueError("infidelity_threshold must lie in (0,1)")
        if render_mode not in (None, "ansi"):
            raise ValueError("only ansi rendering is supported")
        self.registry = registry
        self.builder = observation_builder or MolecularGraphObservationBuilder(registry)
        self.allowed_tasks = allowed_tasks or registry.names
        for name in self.allowed_tasks:
            registry.get(name)
        self.task_probabilities = (
            None if task_probabilities is None else np.asarray(task_probabilities, dtype=np.float64)
        )
        if self.task_probabilities is not None and self.task_probabilities.shape != (
            len(self.allowed_tasks),
        ):
            raise ValueError("task_probabilities shape mismatch")
        self.infidelity_threshold = float(infidelity_threshold)
        self.base_reward = float(base_reward)
        self.overlap_threshold_override = overlap_threshold
        self.apply_bbr = bool(apply_bbr)
        self.render_mode = render_mode
        self._max_steps_config = max_steps
        self._overlap_penalty_config = overlap_penalty

        p = self.builder.padding
        d = registry.feature_dimensions
        self.action_space = spaces.Discrete(p.actions_max)
        self.observation_space = spaces.Dict(
            {
                "atom_features": spaces.Box(-10.0, 10.0, (p.atoms_max, d["atom"]), np.float32),
                "atom_edge_features": spaces.Box(
                    -10.0, 10.0, (p.atom_edges_max, d["atom_edge"]), np.float32
                ),
                "atom_senders": spaces.Box(0, max(p.atoms_max - 1, 0), (p.atom_edges_max,), np.int64),
                "atom_receivers": spaces.Box(0, max(p.atoms_max - 1, 0), (p.atom_edges_max,), np.int64),
                "atom_mask": spaces.Box(0, 1, (p.atoms_max,), np.int8),
                "atom_edge_mask": spaces.Box(0, 1, (p.atom_edges_max,), np.int8),
                "explicit_chemistry_features": spaces.Box(
                    -10.0, 10.0, (d["explicit"],), np.float32
                ),
                "explicit_chemistry_feature_mask": spaces.Box(
                    0.0, 1.0, (d["explicit"],), np.float32
                ),
                "level_features": spaces.Box(
                    -20.0, 20.0, (p.states_max, self.builder.level_input_dim), np.float32
                ),
                "level_feature_mask": spaces.Box(
                    0.0, 1.0, (p.states_max, self.builder.level_input_dim), np.float32
                ),
                "level_edge_features": spaces.Box(
                    -20.0, 20.0, (p.level_edges_max, d["level_edge"]), np.float32
                ),
                "level_senders": spaces.Box(0, max(p.states_max - 1, 0), (p.level_edges_max,), np.int64),
                "level_receivers": spaces.Box(0, max(p.states_max - 1, 0), (p.level_edges_max,), np.int64),
                "level_mask": spaces.Box(0, 1, (p.states_max,), np.int8),
                "level_edge_mask": spaces.Box(0, 1, (p.level_edges_max,), np.int8),
                "global_features": spaces.Box(-20.0, 20.0, (GLOBAL_FEATURE_DIM,), np.float32),
                "pulse_features": spaces.Box(
                    -20.0, 20.0, (p.actions_max, d["pulse"]), np.float32
                ),
                "pulse_feature_mask": spaces.Box(
                    0.0, 1.0, (p.actions_max, d["pulse"]), np.float32
                ),
                "action_mask": spaces.Box(0, 1, (p.actions_max,), np.int8),
                "action_env_indices": spaces.Box(-1, p.actions_max - 1, (p.actions_max,), np.int64),
                "pulse_transition_action": spaces.Box(
                    0, max(p.actions_max - 1, 0), (p.pulse_transitions_max,), np.int64
                ),
                "pulse_transition_src": spaces.Box(
                    0, max(p.states_max - 1, 0), (p.pulse_transitions_max,), np.int64
                ),
                "pulse_transition_dst": spaces.Box(
                    0, max(p.states_max - 1, 0), (p.pulse_transitions_max,), np.int64
                ),
                "pulse_transition_features": spaces.Box(
                    -20.0,
                    20.0,
                    (p.pulse_transitions_max, d["pulse_transition"]),
                    np.float32,
                ),
                "pulse_transition_mask": spaces.Box(
                    0, 1, (p.pulse_transitions_max,), np.int8
                ),
                "task_id": spaces.Box(0, max(registry.by_id), (), np.int64),
            }
        )

        self.task: MoleculeTask | None = None
        self._state: np.ndarray | None = None
        self._step_count = 0

    def _task_max_steps(self, task: MoleculeTask) -> int:
        if isinstance(self._max_steps_config, Mapping):
            return int(self._max_steps_config.get(task.name, 150))
        return int(self._max_steps_config)

    def _task_overlap_penalty(self, task: MoleculeTask) -> float:
        if isinstance(self._overlap_penalty_config, Mapping):
            return float(self._overlap_penalty_config.get(task.name, 0.0))
        return float(self._overlap_penalty_config)

    def max_steps_for(self, task: MoleculeTask | str | int) -> int:
        """Return the configured time limit for a specified molecule task."""
        return self._task_max_steps(self.registry.get(task))

    @property
    def max_steps(self) -> int:
        if self.task is None:
            return max(self._task_max_steps(task) for task in self.registry.tasks)
        return self._task_max_steps(self.task)

    @property
    def confidence_threshold(self) -> float:
        return 1.0 - self.infidelity_threshold

    @property
    def overlap_threshold(self) -> float:
        if self.overlap_threshold_override is not None:
            return float(self.overlap_threshold_override)
        if self.task is None:
            raise RuntimeError("environment has not been reset")
        return 1.0 - 1.0 / self.task.n_states

    @property
    def raw_state(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state.copy()

    def _select_task(self, options: dict[str, Any] | None) -> MoleculeTask:
        if options:
            key = options.get("molecule", options.get("task", options.get("task_id")))
            if key is not None:
                task = self.registry.get(key)
                if task.name not in self.allowed_tasks:
                    raise ValueError(f"task {task.name} is not allowed in this environment")
                return task
        return self.registry.sample(
            self.np_random,
            probabilities=self.task_probabilities,
            allowed=self.allowed_tasks,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.task = self._select_task(options)
        self._state = self.task.model.initial_population.astype(np.float64, copy=True)
        self._step_count = 0
        observation = self.builder.build(
            self.task,
            self._state,
            step_count=0,
            max_steps=self.max_steps,
        )
        return observation, self._info(observation)

    def branch_details(
        self,
        state: np.ndarray,
        action: int,
        *,
        task: MoleculeTask | str | int | None = None,
        next_step_count: int | None = None,
    ) -> MultiBranchDetails:
        active = self.task if task is None else self.registry.get(task)
        if active is None:
            raise RuntimeError("environment has not been reset")
        if not 0 <= int(action) < active.n_actions:
            raise ValueError(f"invalid action {action} for {active.name}")
        branches = active.model.branches(state, int(action), apply_bbr=self.apply_bbr)
        incoming = np.asarray(state, dtype=np.float64)
        incoming = np.clip(incoming, 0.0, None)
        incoming /= incoming.sum()
        dot = branches.next_states @ incoming
        norm = np.linalg.norm(branches.next_states, axis=1) * np.linalg.norm(incoming)
        overlaps = np.divide(dot, norm, out=np.ones(2), where=norm > 0.0)
        rewards = np.full(2, self.base_reward, dtype=np.float64)
        penalty = self._task_overlap_penalty(active)
        if penalty > 0.0:
            threshold = (
                float(self.overlap_threshold_override)
                if self.overlap_threshold_override is not None
                else 1.0 - 1.0 / active.n_states
            )
            rewards -= penalty * (overlaps > threshold)
        terminated = np.max(branches.next_states, axis=1) >= self.confidence_threshold
        del next_step_count  # truncation is returned separately by step().
        return MultiBranchDetails(
            probabilities=branches.probabilities,
            next_states=branches.next_states,
            rewards=rewards,
            terminated=terminated,
            overlaps=overlaps,
        )

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.task is None or self._state is None:
            raise RuntimeError("reset() must be called before step()")
        if not 0 <= int(action) < self.task.n_actions:
            raise ValueError(
                f"action {action} is padded/invalid for {self.task.name}; "
                "respect info['action_mask']"
            )
        details = self.branch_details(self._state, int(action))
        outcome = int(self.np_random.choice(2, p=details.probabilities))
        self._state = details.next_states[outcome].copy()
        self._step_count += 1
        terminated = bool(details.terminated[outcome])
        truncated = bool(self._step_count >= self.max_steps and not terminated)
        observation = self.builder.build(
            self.task,
            self._state,
            step_count=self._step_count,
            max_steps=self.max_steps,
        )
        info = self._info(observation)
        info.update(
            {
                "measurement_outcome": outcome,
                "measurement_probability": float(details.probabilities[outcome]),
                "branch_probabilities": details.probabilities.copy(),
                "all_branch_states": details.next_states.copy(),
                "all_branch_rewards": details.rewards.copy(),
                "overlap": float(details.overlaps[outcome]),
                "pulse_local_index": int(action),
                "pulse_label": self.task.model.action_labels[int(action)],
            }
        )
        return (
            observation,
            float(details.rewards[outcome]),
            terminated,
            truncated,
            info,
        )

    def _info(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        if self.task is None or self._state is None:
            raise RuntimeError("environment has not been reset")
        return {
            "molecule": self.task.name,
            "task_id": self.task.task_id,
            "task_family": self.task.family,
            "transfer_group": self.task.transfer_group,
            "action_mask": observation["action_mask"].copy(),
            "action_env_indices": observation["action_env_indices"].copy(),
            "raw_population": self._state.copy(),
            "confidence": float(np.max(self._state)),
            "most_likely_state": int(np.argmax(self._state)),
            "step_count": self._step_count,
            "max_steps": self.max_steps,
        }

    def render(self) -> str | None:
        if self.task is None or self._state is None:
            return None
        text = (
            f"{self.task.name} step={self._step_count} "
            f"confidence={np.max(self._state):.6f} "
            f"state={np.argmax(self._state)}"
        )
        if self.render_mode == "ansi":
            return text
        return None


__all__ = ["MultiBranchDetails", "MultiMoleculeRLQLSEnv"]
