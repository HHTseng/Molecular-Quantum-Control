#!/usr/bin/env python3
"""Train, transfer, and compare the multi-molecule GNN RL-QLS controller.

Experiment structure
--------------------
1. Pretrain shared parameters on source tasks CaH+ and H3O+.
2. Evaluate zero-shot on held-out related tasks MgH+ and D3O+.
3. Fine-tune the pretrained model on each target for a small budget.
4. Train the same architecture from scratch with the identical target budget.
5. Train a joint all-task reference.
6. Save checkpoints, raw metrics, learning curves, completion curves, chemistry
   embedding similarities, and a Markdown analysis.

The MgH+/D3O+ tasks are transfer-demonstration surrogates; the study validates
software/representation transfer, not molecular spectroscopy accuracy.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlqls.gnn import GNNQConfig  # noqa: E402
from rlqls.multitask import MultiMoleculeRLQLSEnv, build_default_registry  # noqa: E402
from rlqls.multitask.evaluation import (  # noqa: E402
    TaskEvaluationResult,
    chemistry_embeddings,
    evaluate_gnn_batched,
    evaluate_random,
    evaluate_sweeping,
)
from rlqls.multitask.trainer import (  # noqa: E402
    MultiTaskDQNConfig,
    train_multitask_dqn,
)


SOURCE_TASKS = ("CaH+", "H3O+")
# Three CaH+ episodes per H3O+ episode in the supplied reference run.  The
# repeated names control episode scheduling only; replay sampling remains
# approximately balanced across the distinct task queues.
SOURCE_TRAINING_SCHEDULE = ("CaH+", "CaH+", "CaH+", "H3O+")
TARGET_TASKS = ("MgH+", "D3O+")
ALL_TASKS = ("CaH+", "MgH+", "H3O+", "D3O+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-episodes", type=int, default=240)
    parser.add_argument("--target-episodes", type=int, default=60)
    parser.add_argument("--joint-episodes", type=int, default=160)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--max-steps-small", type=int, default=25)
    parser.add_argument("--max-steps-large", type=int, default=20)
    parser.add_argument(
        "--skip-no-chemistry-ablation",
        action="store_true",
        help="Skip the source model with chemistry encoders disabled.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "transfer_study",
    )
    return parser.parse_args()


def moving_mean(values: list[int], window: int = 10) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    output = np.empty_like(array)
    for index in range(array.size):
        start = max(0, index + 1 - window)
        output[index] = np.mean(array[start : index + 1])
    return output


def completion_curve(result: TaskEvaluationResult, max_steps: int) -> np.ndarray:
    return np.asarray(
        [result.completion_fraction(horizon) for horizon in range(1, max_steps + 1)]
    )


def cosine_matrix(embeddings: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(embeddings)
    matrix = np.stack([embeddings[name] for name in names]).astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    return names, matrix @ matrix.T


def make_env(registry, max_steps):
    return MultiMoleculeRLQLSEnv(
        registry,
        max_steps=max_steps,
        overlap_penalty={"H3O+": 1.0, "D3O+": 1.0},
    )


def train_config(
    args,
    *,
    episodes: int,
    tasks: tuple[str, ...],
    seed: int,
    pretrained: bool,
    chemistry: bool = True,
) -> MultiTaskDQNConfig:
    return MultiTaskDQNConfig(
        episodes=episodes,
        training_tasks=tasks,
        learning_rate=1.5e-4 if pretrained else 3e-4,
        gamma=0.95,
        target_tau=2e-3,
        batch_size=args.batch_size,
        warmup_transitions=max(args.batch_size, 16),
        train_every_steps=args.train_every,
        gradient_steps=1,
        epsilon_start=0.30 if pretrained else 1.0,
        epsilon_end=0.03 if pretrained else 0.05,
        epsilon_decay_fraction=0.6,
        seed=seed,
        update_mode="qmdp",
        gnn=GNNQConfig(
            hidden_dim=args.hidden_dim,
            chemistry_context_dim=args.hidden_dim,
            chemistry_layers=1,
            spectroscopy_layers=1,
            use_atomistic_chemistry=chemistry,
            use_explicit_chemistry=chemistry,
        ),
    )


def evaluate_policy_set(
    env,
    network,
    task_name: str,
    *,
    episodes: int,
    seed: int,
    policy_name: str,
):
    return evaluate_gnn_batched(
        env,
        network,
        task_name=task_name,
        episodes=episodes,
        seed=seed,
        policy_name=policy_name,
    )


def result_table_markdown(metrics: dict[str, dict[str, dict]]) -> str:
    lines = [
        "| Target | Policy | Success rate | Mean censored steps | Mean successful steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, policies in metrics.items():
        for policy, summary in policies.items():
            mean_success = summary["mean_successful_length"]
            success_text = f"{100.0 * summary['success_rate']:.1f}%"
            mean_success_text = "n/a" if not math.isfinite(mean_success) else f"{mean_success:.2f}"
            lines.append(
                f"| {target} | {policy} | {success_text} | "
                f"{summary['mean_censored_length']:.2f} | {mean_success_text} |"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = args.output
    checkpoints = output / "checkpoints"
    figures = output / "figures"
    histories_dir = output / "histories"
    for directory in (output, checkpoints, figures, histories_dir):
        directory.mkdir(parents=True, exist_ok=True)

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
    env = make_env(registry, max_steps)

    # 1. Source pretraining.
    source_config = train_config(
        args,
        episodes=args.source_episodes,
        tasks=SOURCE_TRAINING_SCHEDULE,
        seed=args.seed,
        pretrained=False,
    )
    source = train_multitask_dqn(env, source_config)
    source.save(checkpoints / "source_pretrained_CaH_H3O.pt")
    (histories_dir / "source_pretrained.json").write_text(
        json.dumps(source.history.as_dict(), indent=2)
    )

    # 2. Joint all-task reference.
    joint_config = train_config(
        args,
        episodes=args.joint_episodes,
        tasks=ALL_TASKS,
        seed=args.seed + 101,
        pretrained=False,
    )
    joint = train_multitask_dqn(env, joint_config)
    joint.save(checkpoints / "joint_all_tasks.pt")
    (histories_dir / "joint_all_tasks.json").write_text(
        json.dumps(joint.history.as_dict(), indent=2)
    )

    fine_tuned = {}
    scratch = {}
    for target_index, task_name in enumerate(TARGET_TASKS):
        # 3. Few-shot fine-tuning from source pretrained parameters.
        fine_config = train_config(
            args,
            episodes=args.target_episodes,
            tasks=(task_name,),
            seed=args.seed + 1000 + target_index,
            pretrained=True,
        )
        fine = train_multitask_dqn(
            env,
            fine_config,
            initial_online_state=copy.deepcopy(source.online.state_dict()),
            initial_target_state=copy.deepcopy(source.target.state_dict()),
        )
        fine.save(checkpoints / f"fine_tuned_{task_name.replace('+', 'p')}.pt")
        (histories_dir / f"finetune_{task_name.replace('+', 'p')}.json").write_text(
            json.dumps(fine.history.as_dict(), indent=2)
        )
        fine_tuned[task_name] = fine

        # 4. Same target-only budget from random initialization.
        scratch_config = train_config(
            args,
            episodes=args.target_episodes,
            tasks=(task_name,),
            seed=args.seed + 2000 + target_index,
            pretrained=False,
        )
        scratch_run = train_multitask_dqn(env, scratch_config)
        scratch_run.save(checkpoints / f"scratch_{task_name.replace('+', 'p')}.pt")
        (histories_dir / f"scratch_{task_name.replace('+', 'p')}.json").write_text(
            json.dumps(scratch_run.history.as_dict(), indent=2)
        )
        scratch[task_name] = scratch_run

    no_chemistry = None
    if not args.skip_no_chemistry_ablation:
        ablation_config = train_config(
            args,
            episodes=args.source_episodes,
            tasks=SOURCE_TRAINING_SCHEDULE,
            seed=args.seed + 303,
            pretrained=False,
            chemistry=False,
        )
        no_chemistry = train_multitask_dqn(env, ablation_config)
        no_chemistry.save(checkpoints / "source_pretrained_no_chemistry.pt")
        (histories_dir / "source_pretrained_no_chemistry.json").write_text(
            json.dumps(no_chemistry.history.as_dict(), indent=2)
        )

    # 5. Independent policy comparison on held-out targets.
    raw_results: dict[str, dict[str, TaskEvaluationResult]] = {}
    summary: dict[str, dict[str, dict]] = {}
    for target_index, task_name in enumerate(TARGET_TASKS):
        base_seed = args.seed + 10_000 * (target_index + 1)
        policies: dict[str, TaskEvaluationResult] = {
            "zero-shot": evaluate_policy_set(
                env,
                source.online,
                task_name,
                episodes=args.eval_episodes,
                seed=base_seed,
                policy_name="zero-shot",
            ),
            "fine-tuned": evaluate_policy_set(
                env,
                fine_tuned[task_name].online,
                task_name,
                episodes=args.eval_episodes,
                seed=base_seed,
                policy_name="fine-tuned",
            ),
            "scratch": evaluate_policy_set(
                env,
                scratch[task_name].online,
                task_name,
                episodes=args.eval_episodes,
                seed=base_seed,
                policy_name="scratch",
            ),
            "joint": evaluate_policy_set(
                env,
                joint.online,
                task_name,
                episodes=args.eval_episodes,
                seed=base_seed,
                policy_name="joint",
            ),
            "sweeping": evaluate_sweeping(
                env,
                task_name=task_name,
                episodes=args.eval_episodes,
                seed=base_seed,
            ),
            "random": evaluate_random(
                env,
                task_name=task_name,
                episodes=args.eval_episodes,
                seed=base_seed,
            ),
        }
        if no_chemistry is not None:
            policies["zero-shot no chemistry"] = evaluate_policy_set(
                env,
                no_chemistry.online,
                task_name,
                episodes=args.eval_episodes,
                seed=base_seed,
                policy_name="zero-shot no chemistry",
            )
        raw_results[task_name] = policies
        summary[task_name] = {
            name: result.summary((8, 18, 30, 62, max_steps[task_name]))
            for name, result in policies.items()
        }

    # Evaluate source tasks to verify the pretrained controller learned them.
    source_summary = {}
    for index, task_name in enumerate(SOURCE_TASKS):
        result = evaluate_policy_set(
            env,
            source.online,
            task_name,
            episodes=args.eval_episodes,
            seed=args.seed + 50_000 + index,
            policy_name="source-pretrained",
        )
        source_summary[task_name] = result.summary((8, 18, 62, max_steps[task_name]))

    # 6. Figures.
    policy_order = ["zero-shot", "fine-tuned", "scratch", "joint", "sweeping", "random"]
    if no_chemistry is not None:
        policy_order.insert(1, "zero-shot no chemistry")
    x = np.arange(len(TARGET_TASKS), dtype=np.float64)
    width = 0.8 / len(policy_order)

    figure = plt.figure(figsize=(9.0, 5.0))
    axis = figure.add_subplot(111)
    for policy_index, policy in enumerate(policy_order):
        values = [summary[target][policy]["success_rate"] for target in TARGET_TASKS]
        axis.bar(x + (policy_index - (len(policy_order) - 1) / 2) * width, values, width, label=policy)
    axis.set_xticks(x, TARGET_TASKS)
    axis.set_ylabel("Success rate")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Held-out molecule transfer")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "transfer_success_rate.png", dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(9.0, 5.0))
    axis = figure.add_subplot(111)
    for policy_index, policy in enumerate(policy_order):
        values = [summary[target][policy]["mean_censored_length"] for target in TARGET_TASKS]
        axis.bar(x + (policy_index - (len(policy_order) - 1) / 2) * width, values, width, label=policy)
    axis.set_xticks(x, TARGET_TASKS)
    axis.set_ylabel("Mean censored episode length")
    axis.set_title("Transfer-policy pulse cost")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "transfer_mean_censored_steps.png", dpi=180)
    plt.close(figure)

    for task_name in TARGET_TASKS:
        figure = plt.figure(figsize=(7.2, 4.8))
        axis = figure.add_subplot(111)
        axis.plot(
            moving_mean(fine_tuned[task_name].history.episode_lengths),
            label="pretrained fine-tune",
        )
        axis.plot(
            moving_mean(scratch[task_name].history.episode_lengths),
            label="scratch",
        )
        axis.set_xlabel("Target-training episode")
        axis.set_ylabel("Moving mean episode length")
        axis.set_title(f"{task_name}: equal-budget adaptation")
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            figures / f"{task_name.replace('+', 'p')}_adaptation_curve.png",
            dpi=180,
        )
        plt.close(figure)

        figure = plt.figure(figsize=(7.2, 4.8))
        axis = figure.add_subplot(111)
        horizon = np.arange(1, max_steps[task_name] + 1)
        for policy in policy_order:
            result = raw_results[task_name][policy]
            axis.plot(horizon, completion_curve(result, max_steps[task_name]), label=policy)
        axis.set_xlabel("Pulse-measurement steps")
        axis.set_ylabel("Completion fraction")
        axis.set_ylim(0.0, 1.02)
        axis.set_title(f"{task_name}: completion distribution")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(
            figures / f"{task_name.replace('+', 'p')}_completion.png",
            dpi=180,
        )
        plt.close(figure)

    embeddings = chemistry_embeddings(env, source.online, task_names=ALL_TASKS)
    embedding_names, embedding_cosine = cosine_matrix(embeddings)
    figure = plt.figure(figsize=(6.0, 5.2))
    axis = figure.add_subplot(111)
    image = axis.imshow(embedding_cosine, vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(len(embedding_names)), embedding_names, rotation=30, ha="right")
    axis.set_yticks(range(len(embedding_names)), embedding_names)
    axis.set_title("Learned chemistry-embedding cosine similarity")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(figures / "chemistry_embedding_cosine.png", dpi=180)
    plt.close(figure)

    payload = {
        "configuration": vars(args),
        "tasks": {
            task.name: {
                "n_states": task.n_states,
                "n_actions": task.n_actions,
                "family": task.family,
                "transfer_group": task.transfer_group,
                "model_kind": task.model.metadata.get("model_kind"),
                "derived_from": task.model.metadata.get("derived_from"),
            }
            for task in registry.tasks
        },
        "source_training": source.history.task_summary(),
        "joint_training": joint.history.task_summary(),
        "source_evaluation": source_summary,
        "target_evaluation": summary,
        "chemistry_embedding_names": embedding_names,
        "chemistry_embedding_cosine": embedding_cosine.tolist(),
    }
    # Path is not JSON serializable.
    payload["configuration"]["output"] = str(payload["configuration"]["output"])
    (output / "transfer_metrics.json").write_text(json.dumps(payload, indent=2))

    source_lines = "\n".join(
        f"- **{name}:** success {100*value['success_rate']:.1f}%, "
        f"mean censored length {value['mean_censored_length']:.2f}."
        for name, value in source_summary.items()
    )
    transfer_table = result_table_markdown(summary)
    report = rf"""# Multi-Molecule RL-QLS Transfer Study

## Scope

This run validates the integrated Gymnasium environment, the shared chemistry-conditioned GNN, molecule-specific action masks, heterogeneous qMDP replay, and held-out transfer workflow.  It does **not** validate MgH\(^+\) or D\(_3\)O\(^+\) spectroscopy.  Those two tasks are deterministic related-task surrogates derived from the public CaH\(^+\)/H\(_3\)O\(^+\) branch-map reconstructions.

## Experiment

- Source species: `{SOURCE_TASKS}`; episode schedule: `{SOURCE_TRAINING_SCHEDULE}` for {args.source_episodes} total episodes.
- Held-out targets: `{TARGET_TASKS}`.
- Fine-tune and scratch budget: {args.target_episodes} episodes per target.
- Joint all-task reference: {args.joint_episodes} episodes.
- Monte Carlo evaluation: {args.eval_episodes} episodes per target-policy pair.
- Network hidden width: {args.hidden_dim}; expected-branch qMDP Double DQN.

For molecule \(m\), pulse \(a\), and motional outcome \(k\), the environment uses

$$
p_m(k\mid s,a)=\mathbf 1^\mathsf T B_{{a,k}}^{{(m)}}s,
\qquad
F_{{a,k}}^{{(m)}}(s)=\frac{{B_{{a,k}}^{{(m)}}s}}{{\mathbf1^\mathsf T B_{{a,k}}^{{(m)}}s}}.
$$

The shared network scores only valid molecule-local candidates,

$$
a_t=\arg\max_{{a\in\mathcal A_m}}Q_\Theta(o_t^{{(m)}},a),
$$

and qMDP averages both measurement branches in the Bellman target.

## Source-task check

{source_lines}

## Held-out transfer results

{transfer_table}

## Interpretation guide

- **Zero-shot** measures direct transfer before any target gradient update.
- **Fine-tuned** starts from the source-pretrained parameters and receives the same target episode budget as **scratch**.
- **Joint** is an in-distribution reference, because target tasks were included during its training.
- A useful transfer signal is lower pulse cost or higher success for fine-tuning than scratch at equal budget.  Zero-shot performance above random/sweeping indicates that the candidate scorer has learned reusable state/action semantics.
- With small stochastic budgets, differences should be interpreted as implementation diagnostics rather than statistically conclusive learning curves.

## Figures

- `figures/transfer_success_rate.png`
- `figures/transfer_mean_censored_steps.png`
- `figures/MgHp_adaptation_curve.png`
- `figures/D3Op_adaptation_curve.png`
- `figures/MgHp_completion.png`
- `figures/D3Op_completion.png`
- `figures/chemistry_embedding_cosine.png`

## Uncertainties requiring care

1. Pulse descriptors in this implementation are currently derived from the branch matrices \(B_{{a,k}}\).  This is useful for validating the GNN/action machinery but is more informative than primitive experimental descriptors.
2. MgH\(^+\) and D\(_3\)O\(^+\) are related-task deformations, not independently computed Hamiltonian models.
3. The public H\(_3\)O\(^+\) environment remains a surrogate; this run intentionally postpones that separate fidelity question.
4. Chemistry transfer is evaluated with only two source species.  A foundation-model claim would require a much larger and chemically diverse task database plus held-out species splits.
"""
    (output / "TRANSFER_RESULTS.md").write_text(report)
    print(json.dumps(payload["target_evaluation"], indent=2))


if __name__ == "__main__":
    main()
