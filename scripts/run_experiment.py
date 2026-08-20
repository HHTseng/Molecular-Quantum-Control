#!/usr/bin/env python3
"""Train and evaluate one RL-QLS material model.

This script wires together the four layers of the implementation:

1. material physics -> branch matrices B[a,k];
2. branch matrices -> Gymnasium MDP environment;
3. environment -> DQN/qMDP training loop;
4. trained policy -> Monte Carlo evaluation and figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlqls import (  # noqa: E402
    DQNConfig,
    RLQLSEnv,
    build_cah16_surrogate,
    build_h3o130_surrogate,
    train_dqn,
)
from rlqls.evaluation import (  # noqa: E402
    evaluate_network_batched,
    evaluate_sweeping_batched,
)
from rlqls.plotting import plot_completion, plot_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=["cah", "h3o"], required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument(
        "--update-mode",
        choices=["qmdp", "sampled"],
        default="qmdp",
    )
    parser.add_argument(
        "--bootstrap-on-truncation",
        action="store_true",
        help=(
            "Bootstrap across the artificial max-step cutoff.  By default the "
            "cutoff is treated as absorbing failure to preserve reproduction runs."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "manual_run",
    )
    parser.add_argument("--h3o-motional-dim", type=int, default=4)
    parser.add_argument(
        "--h3o-pulse-rule",
        choices=["strongest", "cluster_median"],
        default="strongest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.material == "cah":
        # Physics state dimension N=16; action dimension A=13.
        model = build_cah16_surrogate()
        episodes = args.episodes or 1000
        max_steps = args.max_steps or 100

        # The Gym observation is a 16-component posterior population and the
        # action is one of 13 pulse indices.
        env = RLQLSEnv(model, max_steps=max_steps)

        config = DQNConfig(
            episodes=episodes,
            batch_size=args.batch_size,
            warmup_transitions=128,
            train_every_steps=args.train_every,
            seed=args.seed,
            learning_rate=5e-4,
            target_tau=1e-3,
            gamma=1.0,
            epsilon_end=0.005,
            update_mode=args.update_mode,
            bootstrap_on_truncation=args.bootstrap_on_truncation,
        )
        pulse_counts = (2, 3, 4, 5, 6, 7, 8, 18)

    else:
        # Physics state dimension N=130; inferred action dimension A=218.
        model = build_h3o130_surrogate(
            ROOT / "data",
            motional_dim=args.h3o_motional_dim,
            pulse_rule=args.h3o_pulse_rule,
        )
        episodes = args.episodes or 300
        max_steps = args.max_steps or 150

        # The optional overlap penalty implements the paper's physics-informed
        # discouragement of pulses that barely change the population vector.
        env = RLQLSEnv(
            model,
            max_steps=max_steps,
            overlap_penalty=1.0,
        )

        config = DQNConfig(
            episodes=episodes,
            batch_size=args.batch_size,
            warmup_transitions=256,
            train_every_steps=args.train_every,
            seed=args.seed,
            learning_rate=5e-4,
            target_tau=1e-4,
            gamma=0.9,
            epsilon_end=0.125,
            update_mode=args.update_mode,
            bootstrap_on_truncation=args.bootstrap_on_truncation,
        )
        pulse_counts = (10, 30, 62, 100, 150)

    # Save the finite physics/MDP kernel separately from neural-network weights.
    model.save_npz(args.output / "branch_model.npz")

    trained = train_dqn(env, config)
    trained.save(args.output / "checkpoint.pt")

    # Greedy evaluation: exploration is disabled, so each state chooses
    # argmax_a Q_theta(s,a).  Measurement outcomes remain stochastic.
    rl_result = evaluate_network_batched(
        env,
        trained.online,
        episodes=args.eval_episodes,
        seed=args.seed + 100_000,
    )

    # Fixed reference protocol: cycle through the pulse library in order.
    sweeping_result = evaluate_sweeping_batched(
        env,
        episodes=args.eval_episodes,
        seed=args.seed + 200_000,
    )

    recent_count = min(100, len(trained.history.episode_lengths))
    metrics = {
        "material": args.material,
        "model_metadata": model.metadata,
        "config": vars(args),
        "training": {
            "last100_mean_length": sum(
                trained.history.episode_lengths[-recent_count:]
            )
            / recent_count,
            "last100_success_rate": sum(
                trained.history.episode_success[-recent_count:]
            )
            / recent_count,
            "environment_steps": trained.history.total_environment_steps,
            "optimizer_steps": trained.history.optimizer_steps,
        },
        "rl": rl_result.summary(pulse_counts),
        "sweeping": sweeping_result.summary(pulse_counts),
    }

    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    (args.output / "history.json").write_text(
        json.dumps(trained.history.as_dict())
    )

    plot_training(
        trained.history.episode_lengths,
        args.output / "training.png",
        title=f"{args.material} qMDP training",
    )
    plot_completion(
        {"RL": rl_result, "sweeping": sweeping_result},
        args.output / "completion.png",
        max_steps=max_steps,
        title=f"{args.material} completion",
    )

    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
