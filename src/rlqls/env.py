"""Gymnasium 1.3-compatible RL-QLS environment.

Agent input: molecular posterior population ``s_t``.
Agent output: one pulse index ``a_t``.
Environment observation after action: binary motional outcome ``k_t`` encoded in
``info``; the next RL observation is the conditioned population ``s_(t+1)``.
The measurement and motional reset are fixed parts of every environment step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .compat import gym, spaces
from .model import BranchModel, normalize_probability


@dataclass(frozen=True, slots=True)
class BranchDetails:
    """Physics and control quantities for both ``k=0,1`` outcomes."""

    probabilities: np.ndarray  # (2,): p_k(s,a)
    next_states: np.ndarray  # (2,N_S): s'_{a,k}
    rewards: np.ndarray  # (2,): r_k
    terminated: np.ndarray  # (2,): d_k
    overlaps: np.ndarray  # (2,): cos(s,s'_{a,k})


class RLQLSEnv(gym.Env):
    """Sampled projective-measurement environment (pseudocode Sec. 6).

    The observation ``s_t`` is a molecular population/posterior, action ``a_t``
    selects a pulse, and :meth:`step` samples exactly one Born-rule outcome
    ``k_t ~ Categorical(p_0,p_1)``.
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        model: BranchModel,
        *,
        infidelity_threshold: float = 0.01,
        max_steps: int = 200,
        base_reward: float = -1.0,
        overlap_penalty: float = 0.0,
        overlap_threshold: float | None = None,
        apply_bbr: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < infidelity_threshold < 1.0:
            raise ValueError("infidelity_threshold must lie in (0,1)")
        if max_steps <= 0 or overlap_penalty < 0.0:
            raise ValueError("invalid max_steps/overlap_penalty")
        if render_mode not in {None, "ansi"}:
            raise ValueError("supported render modes: None, 'ansi'")
        self.model = model
        self.infidelity_threshold = float(infidelity_threshold)
        self.max_steps = int(max_steps)
        self.base_reward = float(base_reward)
        self.overlap_penalty = float(overlap_penalty)
        self.overlap_threshold = (
            1.0 - 1.0 / model.n_states if overlap_threshold is None else float(overlap_threshold)
        )
        self.apply_bbr = bool(apply_bbr)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(model.n_actions)
        self.observation_space = spaces.Box(0.0, 1.0, (model.n_states,), dtype=np.float32)
        self._state = model.initial_population.astype(np.float64, copy=True)
        self._step_count = 0
        self._last_outcome: int | None = None

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def confidence_threshold(self) -> float:
        return 1.0 - self.infidelity_threshold

    @property
    def step_count(self) -> int:
        return self._step_count

    def is_terminal(self, state: np.ndarray) -> bool:
        """Test ``max_i s_i >= 1-eta``; this is confidence, not Tr(rho^2)."""
        return bool(float(np.max(state)) >= self.confidence_threshold)

    @staticmethod
    def cosine_overlap(left: np.ndarray, right: np.ndarray) -> float:
        """Return ``o(s,s')=(s^T s')/(||s||_2 ||s'||_2)`` (Sec. 2.7)."""
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 1.0 if denominator <= 0.0 else float(np.dot(left, right) / denominator)

    def reward_for_transition(
        self, old_state: np.ndarray, next_state: np.ndarray
    ) -> tuple[float, float]:
        """Compute ``r_k=r_base-r_o 1[o(s,s'_k)>o_threshold]``."""
        overlap = self.cosine_overlap(old_state, next_state)
        reward = self.base_reward
        # Supplemental Sec. SD gives this trigger and a coefficient r_o but not
        # one unique algebraic reward formula; additive penalty is our assumption.
        if self.overlap_penalty > 0.0 and overlap > self.overlap_threshold:
            reward -= self.overlap_penalty
        return float(reward), overlap

    def branch_details(self, state: np.ndarray, action: int) -> BranchDetails:
        """Deterministically expose all branches needed by the qMDP target."""
        state = normalize_probability(state)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        branch = self.model.branches(state, int(action), apply_bbr=self.apply_bbr)
        rewards = np.empty(2, dtype=np.float64)
        overlaps = np.empty(2, dtype=np.float64)
        terminal = np.empty(2, dtype=np.bool_)
        for k in range(2):
            rewards[k], overlaps[k] = self.reward_for_transition(state, branch.next_states[k])
            terminal[k] = self.is_terminal(branch.next_states[k])
        return BranchDetails(branch.probabilities, branch.next_states, rewards, terminal, overlaps)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to ``s_0`` (normally Boltzmann) or a supplied simplex state."""
        super().reset(seed=seed)
        options = options or {}
        if "initial_state" in options:
            state = normalize_probability(np.asarray(options["initial_state"], dtype=np.float64))
            if state.shape != (self.model.n_states,):
                raise ValueError("initial_state shape mismatch")
        else:
            state = self.model.initial_population.astype(np.float64, copy=True)
        self._state = state
        self._step_count = 0
        self._last_outcome = None
        return self._state.astype(np.float32), {
            "confidence": float(np.max(self._state)),
            "most_likely_state": int(np.argmax(self._state)),
            "step_count": 0,
        }

    def step(self, action: int):
        """Apply pulse ``a``, sample measurement ``k``, and return ``s'_{a,k}``."""
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        old_state = self._state.copy()
        details = self.branch_details(old_state, int(action))

        # Projective measurement: k ~ p_k(s,a); only this branch advances the trajectory.
        outcome = int(self.np_random.choice(2, p=details.probabilities))
        self._state = details.next_states[outcome].copy()
        self._step_count += 1
        self._last_outcome = outcome
        terminated = bool(details.terminated[outcome])
        truncated = bool(self._step_count >= self.max_steps and not terminated)
        # The complete branch set lets qMDP form sum_k p_k[...] despite the sampled step.
        info = {
            "measurement_outcome": outcome,
            "measurement_probability": float(details.probabilities[outcome]),
            "branch_probabilities": details.probabilities.copy(),
            "all_branch_states": details.next_states.astype(np.float32, copy=True),
            "all_branch_rewards": details.rewards.copy(),
            "all_branch_terminated": details.terminated.copy(),
            "all_branch_overlaps": details.overlaps.copy(),
            "confidence": float(np.max(self._state)),
            "most_likely_state": int(np.argmax(self._state)),
            "step_count": self._step_count,
            "action_label": self.model.action_labels[int(action)],
        }
        return (
            self._state.astype(np.float32),
            float(details.rewards[outcome]),
            terminated,
            truncated,
            info,
        )

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        i = int(np.argmax(self._state))
        return (
            f"step={self._step_count}, outcome={self._last_outcome}, "
            f"confidence={self._state[i]:.6f}, state={self.model.state_labels[i]}"
        )


__all__ = ["BranchDetails", "RLQLSEnv"]
