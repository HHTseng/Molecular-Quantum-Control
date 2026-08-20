#!/usr/bin/env python3
"""Print a step-by-step RL/MDP trace for one RL-QLS episode.

This diagnostic makes the hidden transition structure explicit.  At each step
it prints

* the current posterior population s_t;
* the pulse action a_t;
* both probabilities p(k|s_t,a_t);
* both conditional states F_{a_t,k}(s_t);
* the measurement outcome actually sampled by env.step;
* the reward and stopping flags.

Example
-------

    python scripts/trace_one_episode.py --material cah --steps 5 --seed 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlqls import (  # noqa: E402
    QNetwork,
    RLQLSEnv,
    build_cah16_surrogate,
    build_h3o130_surrogate,
)
from rlqls.dqn import greedy_action  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=["cah", "h3o"], default="cah")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--top", type=int, default=5)
    return parser.parse_args()


def top_components(state: np.ndarray, labels: list[str], count: int) -> str:
    indices = np.argsort(state)[::-1][:count]
    return ", ".join(f"{labels[i]}={state[i]:.5f}" for i in indices)


def load_network(
    checkpoint_path: Path,
    n_states: int,
    n_actions: int,
) -> QNetwork:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hidden_sizes = tuple(checkpoint["config"]["hidden_sizes"])
    network = QNetwork(n_states, n_actions, hidden_sizes)
    network.load_state_dict(checkpoint["online"])
    network.eval()
    return network


def main() -> None:
    args = parse_args()
    if args.material == "cah":
        model = build_cah16_surrogate()
        env = RLQLSEnv(model, max_steps=args.steps, render_mode="ansi")
    else:
        model = build_h3o130_surrogate(ROOT / "data")
        env = RLQLSEnv(
            model,
            max_steps=args.steps,
            overlap_penalty=1.0,
            render_mode="ansi",
        )

    network = None
    if args.checkpoint is not None:
        network = load_network(
            args.checkpoint,
            model.n_states,
            model.n_actions,
        )

    rng = np.random.default_rng(args.seed + 1_000_000)
    state, reset_info = env.reset(seed=args.seed)
    print("RESET")
    print(f"  confidence={reset_info['confidence']:.6f}")
    print(f"  top population: {top_components(state, model.state_labels, args.top)}")

    for step in range(args.steps):
        if network is None:
            action = int(rng.integers(model.n_actions))
            policy_name = "random pulse"
        else:
            action = greedy_action(network, state, torch.device("cpu"))
            policy_name = "greedy argmax Q"

        details = env.branch_details(state, action)
        print(f"\nSTEP {step + 1}")
        print(f"  policy={policy_name}")
        print(f"  action a_t={action}: {model.action_labels[action]}")
        for outcome in range(2):
            print(
                f"  branch k={outcome}: p={details.probabilities[outcome]:.6f}, "
                f"reward={details.rewards[outcome]:.3f}, "
                f"terminal={bool(details.terminated[outcome])}"
            )
            print(
                "    top F_(a,k)(s): "
                + top_components(
                    details.next_states[outcome],
                    model.state_labels,
                    args.top,
                )
            )

        next_state, reward, terminated, truncated, info = env.step(action)
        print(
            f"  sampled measurement k_t={info['measurement_outcome']} "
            f"with p={info['measurement_probability']:.6f}"
        )
        print(f"  reward r_(t+1)={reward:.3f}")
        print(f"  next confidence={info['confidence']:.6f}")
        print(
            "  next top population: "
            + top_components(next_state, model.state_labels, args.top)
        )
        print(f"  terminated={terminated}, truncated={truncated}")

        state = next_state
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
