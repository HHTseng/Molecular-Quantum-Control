#!/usr/bin/env python3
"""Train one chemistry-conditioned GNN on an arbitrary molecule subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlqls.gnn import GNNQConfig  # noqa: E402
from rlqls.multitask import MultiMoleculeRLQLSEnv, build_default_registry  # noqa: E402
from rlqls.multitask.trainer import MultiTaskDQNConfig, train_multitask_dqn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["CaH+", "H3O+"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "multitask")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps-small", type=int, default=25)
    parser.add_argument("--max-steps-large", type=int, default=20)
    parser.add_argument("--no-atomistic-chemistry", action="store_true")
    parser.add_argument("--no-explicit-chemistry", action="store_true")
    parser.add_argument("--update-mode", choices=["qmdp", "sampled"], default="qmdp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registry = build_default_registry(
        ROOT / "data",
        precomputed_dir=ROOT / "results",
    )
    max_steps = {
        name: (args.max_steps_small if registry.get(name).n_states <= 32 else args.max_steps_large)
        for name in registry.names
    }
    env = MultiMoleculeRLQLSEnv(
        registry,
        allowed_tasks=tuple(args.tasks),
        max_steps=max_steps,
        overlap_penalty={"H3O+": 1.0, "D3O+": 1.0},
    )
    config = MultiTaskDQNConfig(
        episodes=args.episodes,
        training_tasks=tuple(args.tasks),
        batch_size=args.batch_size,
        warmup_transitions=max(16, args.batch_size),
        train_every_steps=4,
        learning_rate=3e-4,
        gamma=0.95,
        target_tau=2e-3,
        epsilon_end=0.05,
        update_mode=args.update_mode,
        seed=args.seed,
        gnn=GNNQConfig(
            hidden_dim=args.hidden_dim,
            chemistry_context_dim=args.hidden_dim,
            chemistry_layers=1,
            spectroscopy_layers=1,
            use_atomistic_chemistry=not args.no_atomistic_chemistry,
            use_explicit_chemistry=not args.no_explicit_chemistry,
        ),
    )
    trained = train_multitask_dqn(env, config)
    trained.save(args.output / "checkpoint.pt")
    (args.output / "history.json").write_text(
        json.dumps(trained.history.as_dict(), indent=2)
    )
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "tasks": args.tasks,
                "config": {
                    "episodes": args.episodes,
                    "seed": args.seed,
                    "hidden_dim": args.hidden_dim,
                    "update_mode": args.update_mode,
                },
                "task_summary": trained.history.task_summary(),
            },
            indent=2,
        )
    )
    print(json.dumps(trained.history.task_summary(), indent=2))


if __name__ == "__main__":
    main()
