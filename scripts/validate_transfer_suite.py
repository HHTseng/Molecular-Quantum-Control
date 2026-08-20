#!/usr/bin/env python3
r"""Independent multi-checkpoint transfer validation for multi-molecule RL-QLS.

The evaluator launches one isolated worker process per task/policy pair.  This
keeps the peak memory of padded GNN observations bounded and makes the final
comparison reproducible from saved checkpoints without retraining.

The two target tasks shipped with this demonstration, MgH+ and D3O+, are
controlled deformations of the CaH+ and H3O+ branch models.  They validate the
software and representation transfer path; they are not ab-initio molecular
spectroscopy calculations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rlqls.multitask import MultiMoleculeRLQLSEnv, build_default_registry  # noqa: E402
from rlqls.multitask.evaluation import chemistry_embeddings  # noqa: E402
from rlqls.multitask.trainer import load_network_checkpoint  # noqa: E402


TARGETS = ("MgH+", "D3O+")
SOURCES = ("CaH+", "H3O+")
POLICY_ORDER = (
    "zero-shot",
    "zero-shot-no-chemistry",
    "fine-tuned",
    "scratch",
    "joint",
    "sweeping",
    "random",
)


@dataclass(frozen=True)
class PolicySpec:
    label: str
    kind: str
    checkpoint: Path | None


def parse_args() -> argparse.Namespace:
    default = ROOT / "results" / "multimolecule_transfer_final"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed-mgh", type=int, default=9101)
    parser.add_argument("--seed-d3o", type=int, default=9201)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--worker-timeout", type=int, default=180)
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def moving_mean(values: list[int], window: int = 10) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    out = np.empty_like(array)
    for index in range(array.size):
        out[index] = np.mean(array[max(0, index + 1 - window) : index + 1])
    return out


def completion_curve(payload: dict[str, Any]) -> np.ndarray:
    lengths = np.asarray(payload["lengths"], dtype=np.int64)
    success = np.asarray(payload["successes"], dtype=bool)
    max_steps = int(payload["max_steps"])
    return np.asarray(
        [np.mean(success & (lengths <= horizon)) for horizon in range(1, max_steps + 1)],
        dtype=np.float64,
    )


def worker_file(output: Path, task: str, label: str) -> Path:
    safe_task = task.replace("+", "p").replace("3", "3")
    safe_label = label.replace("-", "_").replace(" ", "_")
    return output / "raw" / f"{safe_task}_{safe_label}.json"


def run_worker(
    *,
    output: Path,
    task: str,
    spec: PolicySpec,
    episodes: int,
    seed: int,
    reuse_existing: bool,
    timeout: int,
) -> dict[str, Any]:
    destination = worker_file(output, task, spec.label)
    if reuse_existing and destination.exists():
        return json.loads(destination.read_text())

    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_policy_worker.py"),
        "--task",
        task,
        "--policy",
        spec.kind,
        "--policy-name",
        spec.label,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--output",
        str(destination),
    ]
    if spec.checkpoint is not None:
        command.extend(["--checkpoint", str(spec.checkpoint)])
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        timeout=timeout,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
    )
    return json.loads(destination.read_text())


def checkpoint_history(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False).get("history", {})


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    raw_dir = output / "raw"
    figure_dir = output / "figures"
    checkpoint_dir = output / "checkpoints"
    for directory in (output, raw_dir, figure_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "source": checkpoint_dir / "source_pretrained_CaH_H3O.pt",
        "no_chemistry": checkpoint_dir / "source_pretrained_no_chemistry.pt",
        "fine_MgH+": checkpoint_dir / "fine_tuned_MgHp.pt",
        "scratch_MgH+": checkpoint_dir / "scratch_MgHp.pt",
        "fine_D3O+": checkpoint_dir / "fine_tuned_D3Op.pt",
        "scratch_D3O+": checkpoint_dir / "scratch_D3Op.pt",
        "joint": checkpoint_dir / "joint_all_tasks.pt",
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))

    policy_specs = {
        "MgH+": (
            PolicySpec("zero-shot", "gnn", checkpoints["source"]),
            PolicySpec("zero-shot-no-chemistry", "gnn", checkpoints["no_chemistry"]),
            PolicySpec("fine-tuned", "gnn", checkpoints["fine_MgH+"]),
            PolicySpec("scratch", "gnn", checkpoints["scratch_MgH+"]),
            PolicySpec("joint", "gnn", checkpoints["joint"]),
            PolicySpec("sweeping", "sweeping", None),
            PolicySpec("random", "random", None),
        ),
        "D3O+": (
            PolicySpec("zero-shot", "gnn", checkpoints["source"]),
            PolicySpec("zero-shot-no-chemistry", "gnn", checkpoints["no_chemistry"]),
            PolicySpec("fine-tuned", "gnn", checkpoints["fine_D3O+"]),
            PolicySpec("scratch", "gnn", checkpoints["scratch_D3O+"]),
            PolicySpec("joint", "gnn", checkpoints["joint"]),
            PolicySpec("sweeping", "sweeping", None),
            PolicySpec("random", "random", None),
        ),
    }
    seeds = {"MgH+": args.seed_mgh, "D3O+": args.seed_d3o}

    target_results: dict[str, dict[str, dict[str, Any]]] = {}
    for task in TARGETS:
        target_results[task] = {}
        for spec in policy_specs[task]:
            print(f"Evaluating {task}: {spec.label}", flush=True)
            target_results[task][spec.label] = run_worker(
                output=output,
                task=task,
                spec=spec,
                episodes=args.episodes,
                seed=seeds[task],
                reuse_existing=args.reuse_existing,
                timeout=args.worker_timeout,
            )

    source_results = {}
    for index, task in enumerate(SOURCES):
        source_results[task] = run_worker(
            output=output,
            task=task,
            spec=PolicySpec("source-pretrained", "gnn", checkpoints["source"]),
            episodes=args.episodes,
            seed=9301 + 100 * index,
            reuse_existing=args.reuse_existing,
            timeout=args.worker_timeout,
        )

    # Confidence intervals and condensed summary.
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for task, policies in target_results.items():
        summary[task] = {}
        for label, payload in policies.items():
            base = dict(payload["summary"])
            successes = int(np.sum(payload["successes"]))
            lower, upper = wilson_interval(successes, int(payload["episodes"]))
            base["success_ci95"] = [lower, upper]
            summary[task][label] = base

    # Figure: success rates with Wilson intervals.
    x = np.arange(len(TARGETS), dtype=np.float64)
    width = 0.82 / len(POLICY_ORDER)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for policy_index, label in enumerate(POLICY_ORDER):
        values = np.asarray([summary[task][label]["success_rate"] for task in TARGETS])
        low = np.asarray([summary[task][label]["success_ci95"][0] for task in TARGETS])
        high = np.asarray([summary[task][label]["success_ci95"][1] for task in TARGETS])
        position = x + (policy_index - (len(POLICY_ORDER) - 1) / 2.0) * width
        ax.bar(position, values, width, label=label)
        yerr = np.maximum(np.vstack([values - low, high - values]), 0.0)
        ax.errorbar(position, values, yerr=yerr, fmt="none", capsize=2)
    ax.set_xticks(x, TARGETS)
    ax.set_ylabel("Success probability")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Held-out related-molecule transfer (500 Monte Carlo episodes)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "transfer_success_rate.png", dpi=200)
    plt.close(fig)

    # Figure: censored pulse cost.
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for policy_index, label in enumerate(POLICY_ORDER):
        values = [summary[task][label]["mean_censored_length"] for task in TARGETS]
        position = x + (policy_index - (len(POLICY_ORDER) - 1) / 2.0) * width
        ax.bar(position, values, width, label=label)
    ax.set_xticks(x, TARGETS)
    ax.set_ylabel("Mean pulse-measurement steps (censored)")
    ax.set_title("Pulse cost at the fixed evaluation horizon")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "transfer_mean_censored_steps.png", dpi=200)
    plt.close(fig)

    # Completion curves.
    for task in TARGETS:
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        max_steps = int(next(iter(target_results[task].values()))["max_steps"])
        horizon = np.arange(1, max_steps + 1)
        for label in POLICY_ORDER:
            ax.plot(horizon, completion_curve(target_results[task][label]), label=label)
        ax.set_xlabel("Pulse-measurement steps")
        ax.set_ylabel("Completion fraction")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(f"{task}: completion distribution")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task.replace('+', 'p')}_completion.png", dpi=200)
        plt.close(fig)

    # Equal-budget target adaptation curves.
    for task, fine_key, scratch_key in (
        ("MgH+", "fine_MgH+", "scratch_MgH+"),
        ("D3O+", "fine_D3O+", "scratch_D3O+"),
    ):
        fine_history = checkpoint_history(checkpoints[fine_key])
        scratch_history = checkpoint_history(checkpoints[scratch_key])
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        ax.plot(moving_mean(fine_history.get("episode_lengths", [])), label="pretrained fine-tune")
        ax.plot(moving_mean(scratch_history.get("episode_lengths", [])), label="scratch")
        ax.set_xlabel("Target-training episode")
        ax.set_ylabel("10-episode moving mean length")
        ax.set_title(f"{task}: equal 60-episode adaptation budget")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{task.replace('+', 'p')}_adaptation.png", dpi=200)
        plt.close(fig)

    # Chemistry embedding diagnostic.
    registry = build_default_registry(ROOT / "data", precomputed_dir=ROOT / "results")
    env = MultiMoleculeRLQLSEnv(
        registry,
        max_steps={"CaH+": 25, "MgH+": 25, "H3O+": 20, "D3O+": 20},
        overlap_penalty={"H3O+": 1.0, "D3O+": 1.0},
    )
    source_network, _ = load_network_checkpoint(env, checkpoints["source"])
    embedding = chemistry_embeddings(env, source_network, task_names=registry.names)
    embedding_names = list(embedding)
    matrix = np.stack([embedding[name] for name in embedding_names]).astype(np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    cosine = matrix @ matrix.T
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    image = ax.imshow(cosine, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(embedding_names)), embedding_names, rotation=30, ha="right")
    ax.set_yticks(range(len(embedding_names)), embedding_names)
    ax.set_title("Learned chemistry-context cosine similarity")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "chemistry_embedding_cosine.png", dpi=200)
    plt.close(fig)

    task_metadata = {
        task.name: {
            "n_states": task.n_states,
            "n_actions": task.n_actions,
            "n_pulse_transition_records": task.pulse_library.n_transitions,
            "family": task.family,
            "transfer_group": task.transfer_group,
            "model_kind": task.model.metadata.get("model_kind"),
            "derived_from": task.model.metadata.get("derived_from"),
            "maximum_one_step_conditional_purity": task.maximum_one_step_conditional_purity(),
        }
        for task in registry.tasks
    }
    payload = {
        "evaluation_episodes": args.episodes,
        "seeds": seeds,
        "tasks": task_metadata,
        "source_evaluation": {task: data["summary"] for task, data in source_results.items()},
        "target_evaluation": summary,
        "chemistry_embedding_names": embedding_names,
        "chemistry_embedding_cosine": cosine.tolist(),
        "checkpoints": {key: str(path.relative_to(output)) for key, path in checkpoints.items()},
    }
    (output / "transfer_metrics.json").write_text(json.dumps(payload, indent=2))

    rows = []
    for task in TARGETS:
        for label in POLICY_ORDER:
            item = summary[task][label]
            lo, hi = item["success_ci95"]
            successful = item["mean_successful_length"]
            successful_text = "n/a" if not math.isfinite(successful) else f"{successful:.2f}"
            rows.append(
                f"| {task} | {label} | {100*item['success_rate']:.1f}% "
                f"[{100*lo:.1f}, {100*hi:.1f}] | {item['mean_censored_length']:.2f} | {successful_text} |"
            )

    source_lines = "\n".join(
        f"- **{task}:** success {100*data['summary']['success_rate']:.1f}%, "
        f"mean censored length {data['summary']['mean_censored_length']:.2f}."
        for task, data in source_results.items()
    )
    report = rf"""# Multi-Molecule GNN RL-QLS Transfer Validation

## Scope and status

This study validates the **integrated software architecture**: one shared
chemistry-conditioned GNN, molecule-specific state/action spaces, masked
candidate-pulse scoring, heterogeneous replay, and the expected-branch qMDP
Bellman update.  CaH$^+$ and H$_3$O$^+$ use
the existing reconstruction models.  MgH$^+$ and
D$_3$O$^+$ are controlled related-task deformations used only
to test transfer mechanics; they are not independent spectroscopy predictions.

For molecule $m$, pulse $a$, and motional outcome $k\in\{{0,1\}}$,

$$
q_k^{{(m)}}=B_{{a,k}}^{{(m)}}s,
\qquad
p_m(k\mid s,a)=\mathbf 1^\mathsf Tq_k^{{(m)}},
\qquad
F_{{a,k}}^{{(m)}}(s)=\frac{{q_k^{{(m)}}}}{{p_m(k\mid s,a)}}.
$$

The shared network scores only the local candidate set,

$$
a_t=\arg\max_{{a\in\mathcal A_m}}Q_\Theta(o_t^{{(m)}},a),
$$

and the heterogeneous Double-DQN qMDP target is

$$
y_b=\sum_{{k=0}}^1p_{{b,k}}
\left[r_{{b,k}}+\gamma(1-d_{{b,k}})
Q_{{\bar\Theta}}\!\left(o'_{{b,k}},
\arg\max_{{a'\in\mathcal A_{{m_b}}}}Q_\Theta(o'_{{b,k}},a')\right)\right].
$$

## Training protocol

- Source pretraining: 240 episodes on the round-robin schedule
  `(CaH+, CaH+, CaH+, H3O+)`, giving 180 CaH$^+$ and 60
  H$_3$O$^+$ episodes.
- Held-out target adaptation: 60 episodes per target for both fine-tuning and
  training from scratch.
- Joint reference: 160 episodes, 40 episodes per task.
- Evaluation: {args.episodes} independent Monte Carlo episodes per policy/task;
  brackets below are Wilson 95% intervals for the success probability.

## Source-task check

{source_lines}

## Held-out related-task results

| Target | Policy | Success rate [95% CI] | Mean censored steps | Mean steps among successes |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

## Main observations

1. **MgH$^+$:** source pretraining transfers zero-shot
   ({100*summary['MgH+']['zero-shot']['success_rate']:.1f}%); 60 target episodes
   raise success to {100*summary['MgH+']['fine-tuned']['success_rate']:.1f}%,
   whereas the equal-budget scratch run remains at
   {100*summary['MgH+']['scratch']['success_rate']:.1f}%.  This is the clearest
   positive transfer result in the demonstration.
2. **D$_3$O$^+$:** zero-shot success is
   {100*summary['D3O+']['zero-shot']['success_rate']:.1f}%; fine-tuning reaches
   {100*summary['D3O+']['fine-tuned']['success_rate']:.1f}%, compared with
   {100*summary['D3O+']['scratch']['success_rate']:.1f}% from scratch.  The gain
   is real at this training seed but substantially smaller than for MgH+.
3. The chemistry ablation is **inconclusive**: chemistry conditioning improves
   MgH+ zero-shot performance but not D$_3$O$^+$.  Four tasks and one source
   training seed are insufficient to establish that the learned chemistry
   representation improves generalization.  The chemistry encoder should be
   viewed as implemented infrastructure for a larger molecular database, not a
   validated foundation-model result.
4. Joint training is not the strongest policy under this small, equal-per-task
   budget.  This is compatible with task interference and insufficient updates;
   it does not invalidate the shared-parameter construction.

## Important limitations

- Pulse descriptors currently include summaries derived from the branch maps
  $B_{{a,k}}$.  These are effective for validating variable-action scoring but
  are more informative than primitive experimental pulse descriptors.
- The target tasks preserve source action correspondences by construction.  A
  true cross-species study requires independently calculated energies, Raman
  matrix elements, pulse libraries, detunings, and branch maps.
- These numbers use one training seed.  The 500-episode intervals quantify
  rollout noise, not neural-network training variance.
- The H$_3$O$^+$ branch model remains the earlier reconstruction surrogate; its
  separate physics-fidelity issue is intentionally outside this transfer test.

## Figures

- `figures/transfer_success_rate.png`
- `figures/transfer_mean_censored_steps.png`
- `figures/MgHp_completion.png`
- `figures/D3Op_completion.png`
- `figures/MgHp_adaptation.png`
- `figures/D3Op_adaptation.png`
- `figures/chemistry_embedding_cosine.png`
"""
    (output / "TRANSFER_RESULTS.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
