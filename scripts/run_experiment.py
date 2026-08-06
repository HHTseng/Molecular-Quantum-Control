#!/usr/bin/env python3
"""Run the full workflow: build ``B[a,k]``, train ``Q_theta``, then estimate CDFs."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlqls import RLQLSEnv, DQNConfig, CaH16_surrogate, H3O_130_surrogate, train_dqn
from rlqls.evaluation import evaluate_network_batched, evaluate_sweeping_batched
from rlqls.plotting import plot_completion, plot_training


def parse_args():
    """Parse physical-model, Bellman-update, and Monte Carlo settings."""
    p = argparse.ArgumentParser()
    p.add_argument("--material", choices=["CaH", "H3O"], default="CaH")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--update-mode", choices=["qmdp", "sampled"], default="qmdp")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "manual_run")
    p.add_argument("--H3O-motional-dim", type=int, default=4)
    p.add_argument("--H3O-pulse-rule", choices=["strongest", "cluster_median"], default="strongest")
    return p.parse_args()


def main():
    """Execute pseudocode Secs. 13/14 and evaluate Secs. 15/16."""
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.material == "CaH":
        model = CaH16_surrogate()
        episodes = args.episodes or 1000
        max_steps = args.max_steps or 100
        env = RLQLSEnv(model, max_steps=max_steps)
        cfg = DQNConfig(
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
        )
        pulse_counts = (2, 3, 4, 5, 6, 7, 8, 18)
    else:
        model = H3O_130_surrogate(ROOT / "data", motional_dim=args.H3O_motional_dim, pulse_rule=args.H3O_pulse_rule)
        episodes = args.episodes or 300
        max_steps = args.max_steps or 150
        env = RLQLSEnv(model, max_steps=max_steps, overlap_penalty=1.0)
        cfg = DQNConfig(
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
        )
        pulse_counts = (10, 30, 62, 100, 150)

    # Physics preprocessing artifact: B_{a,k}, s_0, tau_a, and optional T_BBR[a].
    model.save_npz(args.output / "branch_model.npz")
    # Control layer: fit Q_theta(s,a) using the selected Bellman target.
    trained = train_dqn(env, cfg)
    trained.save(args.output / "checkpoint.pt")
    # Greedy testing estimates P(tau_eta<=n); sweeping is the open-loop baseline.
    rl = evaluate_network_batched(
        env, trained.online, episodes=args.eval_episodes, seed=args.seed + 100000
    )
    sweep = evaluate_sweeping_batched(env, episodes=args.eval_episodes, seed=args.seed + 200000)
    metrics = {
        "material": args.material,
        "model_metadata": model.metadata,
        "config": vars(args),
        "training": {
            "last100_mean_length": sum(trained.history.episode_lengths[-100:])
            / min(100, len(trained.history.episode_lengths)),
            "last100_success_rate": sum(trained.history.episode_success[-100:])
            / min(100, len(trained.history.episode_success)),
            "environment_steps": trained.history.total_environment_steps,
            "optimizer_steps": trained.history.optimizer_steps,
        },
        "rl": rl.summary(pulse_counts),
        "sweeping": sweep.summary(pulse_counts),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (args.output / "history.json").write_text(json.dumps(trained.history.as_dict()))
    plot_training(
        trained.history.episode_lengths,
        args.output / "training.png",
        title=f"{args.material} qMDP training",
    )
    plot_completion(
        {"RL": rl, "sweeping": sweep},
        args.output / "completion.png",
        max_steps=max_steps,
        title=f"{args.material} completion",
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
