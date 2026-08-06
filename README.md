# RL-QLS paper implementation

This repository implements the control and reinforcement-learning framework of:

> A. Pipi, X. Tao, A. Wu, P. Narang, and D. R. Leibrandt, **Molecular Quantum Control Algorithm Design by Reinforcement Learning**, arXiv:2410.11839v5 / *Physical Review Research* 8, 033103 (2026).

The implementation is split into two layers:

1. **Paper-faithful control/RL layer**: population-simplex state, discrete pulse actions, binary motional measurement branches, Gymnasium environment, sampled Double DQN, and the expected-branch qMDP Double-DQN target.
2. **Paper-informed physics reconstruction**: approximate CaH+ and H3O+ pulse-conditioned branch matrices, because the complete numerical matrices and the authors' code are not public.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test,physics]'
```

The package pins **Gymnasium 1.3.0**. The optional `physics` extra installs QuTiP for exact propagation when a complete molecular Hamiltonian and pulse library are available.

## One experiment

```bash
python scripts/run_experiment.py \
  --material cah \
  --episodes 1000 \
  --eval-episodes 3000 \
  --seed 7 \
  --output results/my_cah_run
```

A shorter H3O+ run is:

```bash
python scripts/run_experiment.py \
  --material h3o \
  --episodes 100 \
  --eval-episodes 200 \
  --seed 41 \
  --batch-size 32 \
  --train-every 8 \
  --output results/my_h3o_run
```

## Mathematical environment API

For molecular population

$$
 s\in\Delta_{N-1}:=\{x\in\mathbb R_{\geq0}^N:\mathbf 1^T x=1\},
$$

pulse action $a$, and motional outcome $k\in\{0,1\}$, the environment uses

$$
\widetilde s_{a,k}=B_{a,k}s,
\qquad
p(k\mid s,a)=\mathbf 1^T\widetilde s_{a,k},
\qquad
s'_{a,k}=\frac{\widetilde s_{a,k}}{p(k\mid s,a)}.
$$

`RLQLSEnv.step(a)` samples one outcome. `RLQLSEnv.branch_details(s,a)` exposes both branches for the qMDP backup.

The agent controls **only the pulse index**. The projective measurement and motional reset are fixed environment operations after every pulse.

## Code map

- `src/rlqls/model.py`: branch maps $B_{a,k}$ and Boltzmann initialization.
- `src/rlqls/env.py`: Gymnasium environment.
- `src/rlqls/dqn.py`: DQN, Double DQN, replay, target network, and qMDP expected backup.
- `src/rlqls/materials.py`: CaH+ and H3O+ reconstructions.
- `src/rlqls/qutip_builder.py`: generic exact QuTiP branch-matrix builder.
- `src/rlqls/bbr.py`: blackbody-radiation rate generators and propagators.
- `src/rlqls/evaluation.py`: sequential and batched evaluation.
- `results/reproduction_report.md`: detailed numerical assessment and limitations.

## Verification

```bash
PYTHONPATH=src pytest -q tests
```

The current tests verify branch-map trace preservation, Gymnasium return signatures, and the required CaH+/H3O+ state-action dimensions.

## Most important limitation

The paper does not publish the exact pulse-conditioned transition matrices. Consequently, the CaH+ result is a close reduced-model reproduction, while the H3O+ result is only a partial sensitivity study. See the report before interpreting any numerical metric.
