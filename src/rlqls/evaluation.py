"""Monte Carlo policy evaluation and reference policies (pseudocode Secs. 15--16)."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable
import numpy as np, torch
from .dqn import QNetwork, greedy_action
from .env import RLQLSEnv


@dataclass(slots=True)
class EvaluationResult:
    """Episode stopping times, returns, and terminal molecular-state estimates."""
    lengths: list[int]
    returns: list[float]
    successes: list[bool]
    terminal_states: list[int]

    @property
    def success_rate(self):
        """Estimate ``P(tau_eta <= max_steps)`` from completed episodes."""
        return float(np.mean(self.successes))

    @property
    def mean_successful_length(self):
        """Estimate ``E[tau_eta | tau_eta <= max_steps]``."""
        x = [n for n, ok in zip(self.lengths, self.successes) if ok]
        return float(np.mean(x)) if x else float("nan")

    def completion_fraction(self, n):
        """Empirical CDF ``P_hat(tau_eta <= n)`` of preparation pulse count."""
        return float(
            np.mean([ok and length <= n for length, ok in zip(self.lengths, self.successes)])
        )

    def summary(self, pulse_counts=()):
        out = {
            "episodes": len(self.lengths),
            "success_rate": self.success_rate,
            "mean_successful_length": self.mean_successful_length,
            "mean_censored_length": float(np.mean(self.lengths)),
            "median_censored_length": float(np.median(self.lengths)),
        }
        if pulse_counts:
            out["completion_fraction"] = {str(n): self.completion_fraction(n) for n in pulse_counts}
        return out

    def as_dict(self):
        return asdict(self)


def evaluate_policy(
    env: RLQLSEnv, policy: Callable[[np.ndarray, int], int], *, episodes: int, seed: int
):
    """Roll out a policy until confidence termination or time-limit censoring."""
    lengths = []
    returns = []
    successes = []
    terminal = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = truncated = False
        step = 0
        ret = 0.0
        info = {"most_likely_state": int(np.argmax(state))}
        while not (done or truncated):
            state, r, done, truncated, info = env.step(int(policy(state, step)))
            ret += r
            step += 1
        lengths.append(step)
        returns.append(ret)
        successes.append(bool(done))
        terminal.append(int(info["most_likely_state"]))
    return EvaluationResult(lengths, returns, successes, terminal)


def evaluate_network(env, network: QNetwork, *, episodes: int, seed: int, device="cpu"):
    """Evaluate the greedy learned policy ``pi(s)=argmax_a Q_theta(s,a)``."""
    network.eval()
    d = torch.device(device)
    return evaluate_policy(
        env, lambda s, _: greedy_action(network, s, d), episodes=episodes, seed=seed
    )


def evaluate_sweeping(env, *, episodes: int, seed: int, action_order=None):
    """Evaluate the open-loop baseline ``a_t=order[t mod N_A]`` (Sec. 16)."""
    order = action_order or list(
        env.model.metadata.get("sweeping_order", range(env.model.n_actions))
    )
    return evaluate_policy(env, lambda _s, t: order[t % len(order)], episodes=episodes, seed=seed)


def information_gain_action(env: RLQLSEnv, state: np.ndarray) -> int:
    """Choose ``argmin_a sum_k p_k H(s'_{a,k})`` as a diagnostic policy.

    Here ``H(s)=-sum_i s_i log(s_i)``.  Since ``H(s)`` is fixed before the
    action, minimizing expected posterior entropy maximizes information gain.
    """
    states = np.repeat(np.asarray(state, dtype=np.float64)[None, :], env.model.n_actions, axis=0)
    actions = np.arange(env.model.n_actions)
    p, next_states = env.model.batch_branches(states, actions, apply_bbr=env.apply_bbr)
    entropy = -np.sum(
        np.where(next_states > 0, next_states * np.log(next_states + 1e-300), 0.0), axis=2
    )
    expected = np.sum(p * entropy, axis=1)  # E[H(s')|s,a]=sum_k p_k H(s'_k)
    return int(np.argmin(expected))


def evaluate_information_gain(env, *, episodes: int, seed: int):
    """Evaluate the model-based entropy-reduction reference policy."""
    return evaluate_policy(
        env, lambda s, _: information_gain_action(env, s), episodes=episodes, seed=seed
    )


__all__ = [
    "EvaluationResult",
    "evaluate_policy",
    "evaluate_network",
    "evaluate_sweeping",
    "evaluate_information_gain",
]


def _batched_rollout(
    env: RLQLSEnv,
    action_batch_fn,
    *,
    episodes: int,
    seed: int,
) -> EvaluationResult:
    """Vectorized Monte Carlo evaluation for identical initial populations.

    Each row is an independent trajectory.  Outcome ``k`` is sampled from
    ``p_k(s,a)`` and completion occurs at the stopping time
    ``tau_eta=inf{t: max_i s_(t,i)>=1-eta}``.
    """
    rng = np.random.default_rng(seed)
    states = np.repeat(env.model.initial_population[None, :], episodes, axis=0).astype(np.float64)
    active = np.ones(episodes, dtype=bool)
    lengths = np.full(episodes, env.max_steps, dtype=np.int64)
    returns = np.zeros(episodes, dtype=np.float64)
    successes = np.zeros(episodes, dtype=bool)
    terminal_states = np.argmax(states, axis=1).astype(np.int64)
    for step in range(env.max_steps):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        current = states[indices]
        actions = np.asarray(action_batch_fn(current, step), dtype=np.int64)
        if actions.shape != (indices.size,):
            raise ValueError("batched policy must return shape (active_episodes,)")
        probabilities, branches = env.model.batch_branches(
            current, actions, apply_bbr=env.apply_bbr
        )
        # Inverse-CDF sampling for binary k: k=0 iff u<p_0, otherwise k=1.
        outcomes = (rng.random(indices.size) >= probabilities[:, 0]).astype(np.int64)
        selected = branches[np.arange(indices.size), outcomes]

        dot = np.sum(current * selected, axis=1)
        norm = np.linalg.norm(current, axis=1) * np.linalg.norm(selected, axis=1)
        overlap = np.divide(dot, norm, out=np.ones_like(dot), where=norm > 0.0)
        reward = np.full(indices.size, env.base_reward, dtype=np.float64)
        if env.overlap_penalty > 0.0:
            reward -= env.overlap_penalty * (overlap > env.overlap_threshold)
        returns[indices] += reward
        states[indices] = selected
        terminal_states[indices] = np.argmax(selected, axis=1)
        finished = np.max(selected, axis=1) >= env.confidence_threshold
        if np.any(finished):
            done_indices = indices[finished]
            successes[done_indices] = True
            lengths[done_indices] = step + 1
            active[done_indices] = False
    return EvaluationResult(
        lengths.tolist(), returns.tolist(), successes.tolist(), terminal_states.tolist()
    )


def evaluate_network_batched(
    env: RLQLSEnv,
    network: QNetwork,
    *,
    episodes: int,
    seed: int,
    device: str = "cpu",
) -> EvaluationResult:
    """Batched evaluation of ``argmax_a Q_theta(s,a)``."""
    network.eval()
    torch_device = torch.device(device)

    def policy(states: np.ndarray, _step: int) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(states, dtype=torch.float32, device=torch_device)
            return network(tensor).argmax(dim=1).cpu().numpy()

    return _batched_rollout(env, policy, episodes=episodes, seed=seed)


def evaluate_sweeping_batched(
    env: RLQLSEnv,
    *,
    episodes: int,
    seed: int,
    action_order: list[int] | None = None,
) -> EvaluationResult:
    """Batched evaluation of the cyclic open-loop pulse sequence."""
    order = action_order or list(
        env.model.metadata.get("sweeping_order", range(env.model.n_actions))
    )
    return _batched_rollout(
        env,
        lambda states, step: np.full(states.shape[0], order[step % len(order)], dtype=np.int64),
        episodes=episodes,
        seed=seed,
    )


__all__ += ["evaluate_network_batched", "evaluate_sweeping_batched"]
