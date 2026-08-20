"""Gymnasium environment for reinforcement-learning quantum-logic spectroscopy.

The environment implements one complete experimental control step:

    posterior population s_t
        -> agent selects pulse a_t
        -> coherent pulse evolution U_{a_t}
        -> projective motional measurement k_t in {0,1}
        -> conditional population update s_{t+1}
        -> motional cooling/reset.

Only the pulse index is controlled by the agent.  The projective measurement,
its stochastic outcome, and the motional reset are fixed parts of the
environment transition.

Gym observation
---------------
The observation is the normalized population vector

    s_t = (P_{t,1}, ..., P_{t,N})^T,

not the raw measurement bit k_t.  In an ideal experiment, s_t is obtained by
filtering the complete pulse/outcome history.  The current measurement outcome
is nevertheless returned in ``info["measurement_outcome"]`` for diagnostics.

Gym action
----------
An integer ``a_t`` in ``Discrete(N_A)`` selects one precomputed Raman
blue-sideband pulse from the pulse library.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .compat import gym, spaces
from .model import BranchModel, normalize_probability


@dataclass(frozen=True, slots=True)
class BranchDetails:
    """Complete one-step transition kernel for a fixed ``(state, action)``."""

    probabilities: np.ndarray  # p(k|s,a), shape (2,)
    next_states: np.ndarray  # F_{a,k}(s), shape (2,N)
    rewards: np.ndarray  # r(s,a,k), shape (2,)
    terminated: np.ndarray  # terminal flag for each branch, shape (2,)
    overlaps: np.ndarray  # cosine overlap o(s,F_{a,k}(s)), shape (2,)


class RLQLSEnv(gym.Env):
    """Finite continuous-state, discrete-action RL-QLS environment.

    The state space is the probability simplex embedded in ``Box(0,1,(N,))``.
    Gymnasium has no native simplex space, so the stronger constraint
    ``sum_i s_i = 1`` is enforced internally by :func:`normalize_probability`.
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

        # Supplemental Sec. SD uses the trigger
        # o(s_t,s_{t+1}) > 1 - 1/N_S.  The exact additive reward formula is not
        # printed, so the implementation below is a documented assumption.
        self.overlap_threshold = (
            1.0 - 1.0 / model.n_states
            if overlap_threshold is None
            else float(overlap_threshold)
        )
        self.apply_bbr = bool(apply_bbr)
        self.render_mode = render_mode

        # Action a is a pulse-library index.
        self.action_space = spaces.Discrete(model.n_actions)

        # Observation s is a molecular population vector.  The Box describes
        # component bounds; normalization to the simplex is enforced by code.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(model.n_states,),
            dtype=np.float32,
        )

        self._state = model.initial_population.astype(np.float64, copy=True)
        self._step_count = 0
        self._last_outcome: int | None = None

    @property
    def state(self) -> np.ndarray:
        """Defensive copy of the current posterior population."""

        return self._state.copy()

    @property
    def confidence_threshold(self) -> float:
        """Terminal confidence ``1 - eta``."""

        return 1.0 - self.infidelity_threshold

    @property
    def step_count(self) -> int:
        return self._step_count

    def is_terminal(self, state: np.ndarray) -> bool:
        """Return true when one molecular eigenstate has posterior >= 1-eta."""

        return bool(float(np.max(state)) >= self.confidence_threshold)

    @staticmethod
    def cosine_overlap(left: np.ndarray, right: np.ndarray) -> float:
        """Physics-informed similarity used to penalize ineffective pulses."""

        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return 1.0 if denominator <= 0.0 else float(np.dot(left, right) / denominator)

    def reward_for_transition(
        self,
        old_state: np.ndarray,
        next_state: np.ndarray,
    ) -> tuple[float, float]:
        """Evaluate the branch reward.

        Base objective:

            r = -1 per pulse/measurement step,

        so maximizing return is equivalent to minimizing expected preparation
        time.  For large systems, an optional extra penalty discourages pulses
        whose posterior state is almost unchanged.
        """

        overlap = self.cosine_overlap(old_state, next_state)
        reward = self.base_reward
        if self.overlap_penalty > 0.0 and overlap > self.overlap_threshold:
            reward -= self.overlap_penalty
        return float(reward), overlap

    def branch_details(self, state: np.ndarray, action: int) -> BranchDetails:
        """Return both possible measurement branches without sampling.

        This is the model-based extension required by the paper's qMDP update.
        A conventional Gym environment normally exposes only the realized next
        state.  Here DQN training may also query the complete known quantum
        measurement kernel to compute an exact expected Bellman target.
        """

        state = normalize_probability(state)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        branch = self.model.branches(state, int(action), apply_bbr=self.apply_bbr)
        rewards = np.empty(2, dtype=np.float64)
        overlaps = np.empty(2, dtype=np.float64)
        terminal = np.empty(2, dtype=np.bool_)

        for k in range(2):
            rewards[k], overlaps[k] = self.reward_for_transition(
                state,
                branch.next_states[k],
            )
            terminal[k] = self.is_terminal(branch.next_states[k])

        return BranchDetails(
            probabilities=branch.probabilities,
            next_states=branch.next_states,
            rewards=rewards,
            terminated=terminal,
            overlaps=overlaps,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode from the thermal population or a supplied state."""

        # Gymnasium's base reset initializes self.np_random from ``seed``.
        super().reset(seed=seed)
        options = options or {}

        if "initial_state" in options:
            state = normalize_probability(
                np.asarray(options["initial_state"], dtype=np.float64)
            )
            if state.shape != (self.model.n_states,):
                raise ValueError("initial_state shape mismatch")
        else:
            state = self.model.initial_population.astype(np.float64, copy=True)

        self._state = state
        self._step_count = 0
        self._last_outcome = None

        observation = self._state.astype(np.float32)
        info = {
            "confidence": float(np.max(self._state)),
            "most_likely_state": int(np.argmax(self._state)),
            "step_count": 0,
        }
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one pulse, sample its motional readout, and condition the state.

        The stochastic transition is

            k_t ~ p(. | s_t,a_t),
            s_{t+1} = F_{a_t,k_t}(s_t).

        Returns follow the Gymnasium API:

            observation, reward, terminated, truncated, info.

        ``terminated`` means physical task success.  ``truncated`` means the
        artificial simulation time limit ``max_steps`` was reached.
        """

        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")

        old_state = self._state.copy()
        details = self.branch_details(old_state, int(action))

        # The agent controls the distribution through the pulse, but cannot
        # choose the projective measurement result itself.
        outcome = int(self.np_random.choice(2, p=details.probabilities))
        self._state = details.next_states[outcome].copy()
        self._step_count += 1
        self._last_outcome = outcome

        terminated = bool(details.terminated[outcome])
        truncated = bool(self._step_count >= self.max_steps and not terminated)

        info = {
            # Realized experimental observation.
            "measurement_outcome": outcome,
            "measurement_probability": float(details.probabilities[outcome]),
            # Complete model branches, useful for diagnostics and qMDP checks.
            "branch_probabilities": details.probabilities.copy(),
            "all_branch_states": details.next_states.astype(np.float32, copy=True),
            "all_branch_rewards": details.rewards.copy(),
            "all_branch_terminated": details.terminated.copy(),
            "all_branch_overlaps": details.overlaps.copy(),
            # Posterior summary.
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
        """Return a compact text rendering of the posterior state."""

        if self.render_mode != "ansi":
            return None
        i = int(np.argmax(self._state))
        return (
            f"step={self._step_count}, outcome={self._last_outcome}, "
            f"confidence={self._state[i]:.6f}, state={self.model.state_labels[i]}"
        )


__all__ = ["BranchDetails", "RLQLSEnv"]
