#!/usr/bin/env python3
"""Evaluate one policy/task pair in an isolated process.

This small worker is used by ``validate_transfer_suite.py``.  Process isolation
keeps the peak memory of padded large-molecule GNN batches predictable when many
checkpoints are compared in one study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy", choices=("gnn", "random", "sweeping"), required=True)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--max-steps-small", type=int, default=25)
    parser.add_argument("--max-steps-large", type=int, default=20)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_default_registry(
        ROOT / "data",
        precomputed_dir=ROOT / "results",
    )
    max_steps = {
        name: (
            args.max_steps_small
            if registry.get(name).n_states <= 32
            else args.max_steps_large
        )
        for name in registry.names
    }
    env = MultiMoleculeRLQLSEnv(
        registry,
        max_steps=max_steps,
        overlap_penalty={"H3O+": 1.0, "D3O+": 1.0},
    )

    if args.policy == "gnn":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --policy gnn")
        network, checkpoint = load_network_checkpoint(env, args.checkpoint)
        result = evaluate_gnn_batched(
            env,
            network,
            task_name=args.task,
            episodes=args.episodes,
            seed=args.seed,
            policy_name=args.policy_name,
            inference_batch_size=args.inference_batch_size,
            torch_num_threads=1,
        )
        checkpoint_training_tasks = checkpoint.get("config", {}).get("training_tasks")
    elif args.policy == "random":
        result = evaluate_random(
            env,
            task_name=args.task,
            episodes=args.episodes,
            seed=args.seed,
        )
        checkpoint_training_tasks = None
    else:
        result = evaluate_sweeping(
            env,
            task_name=args.task,
            episodes=args.episodes,
            seed=args.seed,
        )
        checkpoint_training_tasks = None

    horizons = sorted(set((2, 8, 18, max_steps[args.task])))
    payload = {
        "task": args.task,
        "policy": args.policy_name,
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "checkpoint_training_tasks": checkpoint_training_tasks,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_steps": max_steps[args.task],
        "summary": result.summary(tuple(horizons)),
        "lengths": result.lengths,
        "returns": result.returns,
        "successes": result.successes,
        "terminal_states": result.terminal_states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
