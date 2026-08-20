#!/usr/bin/env python3
"""Independently evaluate a saved shared-GNN checkpoint on any molecule task.

This script never trains.  It reconstructs the registry/environment, loads the
checkpoint, and compares the learned policy with random and pulse-sweeping
baselines.  It is suitable for held-out zero-shot transfer validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlqls.multitask import MultiMoleculeRLQLSEnv, build_default_registry  # noqa: E402
from rlqls.multitask.evaluation import (  # noqa: E402
    evaluate_gnn_batched,
    evaluate_random,
    evaluate_sweeping,
)
from rlqls.multitask.trainer import load_network_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["MgH+", "D3O+"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "transfer_eval")
    parser.add_argument("--max-steps-small", type=int, default=50)
    parser.add_argument("--max-steps-large", type=int, default=100)
    return parser.parse_args()


def completion_curve(lengths: list[int], successes: list[bool], max_steps: int) -> np.ndarray:
    return np.asarray(
        [
            np.mean(
                [success and length <= horizon for length, success in zip(lengths, successes)]
            )
            for horizon in range(1, max_steps + 1)
        ]
    )


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
        max_steps=max_steps,
        overlap_penalty={"H3O+": 1.0, "D3O+": 1.0},
    )
    network, checkpoint = load_network_checkpoint(env, args.checkpoint)

    metrics: dict[str, dict] = {}
    for offset, task_name in enumerate(args.tasks):
        learned = evaluate_gnn_batched(
            env,
            network,
            task_name=task_name,
            episodes=args.episodes,
            seed=args.seed + 1000 * offset,
            policy_name="checkpoint",
        )
        random = evaluate_random(
            env,
            task_name=task_name,
            episodes=args.episodes,
            seed=args.seed + 1000 * offset + 1,
        )
        sweeping = evaluate_sweeping(
            env,
            task_name=task_name,
            episodes=args.episodes,
            seed=args.seed + 1000 * offset + 2,
        )
        results = [learned, sweeping, random]
        metrics[task_name] = {
            result.policy_name: result.summary((8, 18, 30, 62, max_steps[task_name]))
            for result in results
        }

        figure = plt.figure(figsize=(7.2, 4.8))
        axis = figure.add_subplot(111)
        horizon = np.arange(1, max_steps[task_name] + 1)
        for result in results:
            axis.plot(
                horizon,
                completion_curve(result.lengths, result.successes, max_steps[task_name]),
                label=result.policy_name,
            )
        axis.set_xlabel("Pulse-measurement steps")
        axis.set_ylabel("Completion fraction")
        axis.set_ylim(0.0, 1.02)
        axis.set_title(f"{task_name}: independent checkpoint evaluation")
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output / f"{task_name.replace('+', 'p')}_completion.png", dpi=180)
        plt.close(figure)

    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_training_tasks": checkpoint.get("config", {}).get("training_tasks"),
        "evaluation_episodes": args.episodes,
        "metrics": metrics,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
