# Multi-Molecule GNN RL-QLS

This repository implements and extends the control framework of:

> A. Pipi, X. Tao, A. Wu, P. Narang, and D. R. Leibrandt,
> **Molecular Quantum Control Algorithm Design by Reinforcement Learning**,
> arXiv:2410.11839v5 / *Physical Review Research* **8**, 033103 (2026).

It contains two compatible layers:

1. the original single-molecule Gymnasium/Double-DQN/qMDP reconstruction for
   CaH$^+$ or H$_3$O$^+$;
2. a new multi-task extension in which molecule-specific MDPs share one
   chemistry-conditioned, variable-action GNN Q-function.

The multi-task implementation uses one molecule per episode. It does not assume
that several molecular ions coherently share one motional mode in the same
experimental shot.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test,physics]'
```

The core requirements include Gymnasium 1.3, NumPy, SciPy, PyTorch, and
Matplotlib. The optional `physics` extra installs QuTiP.

## Multi-molecule task family

For molecule $m$, the state and local action set are

$$
s^{(m)}\in\Delta_{N_m-1},
\qquad
\mathcal A_m=\{0,\ldots,A_m-1\}.
$$

The quantum transition model remains molecule specific:

$$
q_k^{(m)}=B_{a,k}^{(m)}s,
\qquad
p_m(k\mid s,a)=\mathbf 1^{\mathsf T}q_k^{(m)},
\qquad
s'_k=\frac{q_k^{(m)}}{p_m(k\mid s,a)}.
$$

One shared GNN scores each molecule's valid pulse candidates:

$$
Q_\Theta(o_m,a),
\qquad a\in\mathcal A_m.
$$

Padding provides fixed tensors for batching; an action mask ensures that the
agent never selects pulse slots outside the local library.

The default demonstration registry contains:

| task | states | actions | status |
|---|---:|---:|---|
| CaH$^+$ | 16 | 13 | existing reconstruction, source |
| MgH$^+$ | 16 | 13 | related-task transfer surrogate |
| H$_3$O$^+$ | 130 | 218 | existing reconstruction, source |
| D$_3$O$^+$ | 130 | 218 | related isotopologue transfer surrogate |

MgH$^+$ and D$_3$O$^+$ validate the transfer software path. They are not
independently calculated molecular spectra.

## Main commands

Inspect the integrated environment:

```bash
PYTHONPATH=src python scripts/train_multitask.py --help
```

Train a shared controller:

```bash
PYTHONPATH=src python scripts/train_multitask.py \
  --tasks 'CaH+' 'H3O+' \
  --episodes 240 \
  --output results/source_controller
```

Run the complete pretrain/zero-shot/fine-tune/scratch workflow:

```bash
PYTHONPATH=src python scripts/run_transfer_study.py \
  --source-episodes 240 \
  --target-episodes 60 \
  --joint-episodes 160 \
  --eval-episodes 500
```

Independently evaluate a saved checkpoint:

```bash
PYTHONPATH=src python scripts/evaluate_transfer.py \
  --checkpoint results/source_controller/checkpoint.pt \
  --tasks 'MgH+' 'D3O+' \
  --episodes 500
```

Rebuild the supplied multi-policy transfer comparison from saved checkpoints:

```bash
PYTHONPATH=src python scripts/validate_transfer_suite.py \
  --output results/multimolecule_transfer_final \
  --episodes 500 \
  --reuse-existing
```

## Architecture map

### Physics and Gym environment

- `src/rlqls/model.py`: molecule-specific branch maps $B_{a,k}$.
- `src/rlqls/multitask/task.py`: one molecular MDP plus transferable metadata.
- `src/rlqls/multitask/registry.py`: extensible molecule registry.
- `src/rlqls/multitask/env.py`: integrated Gymnasium environment and masks.
- `src/rlqls/multitask/builders.py`: CaH$^+$/H$_3$O$^+$ sources and related targets.

### Graph observation and chemistry

- `src/rlqls/features/chemistry.py`: atom/isotope chemistry graphs.
- `src/rlqls/features/spectroscopy.py`: level and pulse-transition descriptors.
- `src/rlqls/multitask/observation.py`: padded masked graph observations.

### Shared GNN and learning

- `src/rlqls/gnn/chemistry_encoder.py`: atom/isotope message passing.
- `src/rlqls/gnn/spectroscopy_encoder.py`: chemistry-conditioned level GNN.
- `src/rlqls/gnn/pulse_scorer.py`: variable-cardinality pulse/hyperedge scorer.
- `src/rlqls/gnn/q_network.py`: shared $Q_\Theta(o_m,a)$.
- `src/rlqls/multitask/replay.py`: task-balanced replay.
- `src/rlqls/multitask/qmdp.py`: heterogeneous expected-branch target.
- `src/rlqls/multitask/trainer.py`: shared Double-DQN/qMDP training.

## Walkthrough and results

- [`MULTI_MOLECULE_GNN_RLQLS_WALKTHROUGH.md`](MULTI_MOLECULE_GNN_RLQLS_WALKTHROUGH.md): detailed code and mathematical walkthrough.
- [`results/multimolecule_transfer_final/TRANSFER_RESULTS.md`](results/multimolecule_transfer_final/TRANSFER_RESULTS.md): supplied transfer experiment and limitations.
- [`RL_MDP_WORKFLOW_WALKTHROUGH.md`](RL_MDP_WORKFLOW_WALKTHROUGH.md): original single-molecule workflow.
- [`results/reproduction_report.md`](results/reproduction_report.md): original CaH$^+$/H$_3$O$^+$ reconstruction assessment.

## Verification

```bash
PYTHONPATH=src pytest -q
```

The tests cover branch-map normalization, Gymnasium signatures, variable
state/action sizes, action masking, shared GNN forward passes, multi-task qMDP
training, and a terminal-reachability diagnostic for the related surrogates.

## Interpretation limits

The exact pulse-conditioned matrices used by the paper are not public. The
provided source environments therefore remain reconstructions. The new
multi-molecule transfer experiment validates the architecture and learning
workflow; it is not evidence for quantitative MgH$^+$ or D$_3$O$^+$ control.
