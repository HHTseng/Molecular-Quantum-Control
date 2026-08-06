# RL-QLS: reinforcement learning for molecular quantum control

Implementation of A. Pipi *et al.*, **Molecular Quantum Control Algorithm Design by Reinforcement Learning**, arXiv:2410.11839v5 / *Physical Review Research* **8**, 033103 (2026).

The repository separates:

- the control layer: population-state Gymnasium environment, sampled Double DQN, and expected-branch quantum-MDP (qMDP) Double DQN;
- the physics layer: reconstructed CaH$^+$ and H$_3$O$^+$ pulse maps. These are approximate because the paper does not publish its full Hamiltonians or pulse-conditioned matrices.

## Physics model

### Molecular state

After each measurement/cooling cycle, coherences are discarded and the molecular state is represented by populations

$$
s=(s_1,\ldots,s_{N_S})^T\in\Delta_{N_S-1},
\qquad s_i\geq0,
\qquad \mathbf 1^Ts=1.
$$

The default initial state is thermal:

$$
s_{0,i}=\frac{e^{-E_i/(k_BT)}}{Z},
\qquad
Z=\sum_j e^{-E_j/(k_BT)}.
$$

### Pulse and projective measurement

Action $a\in\{0,\ldots,N_A-1\}$ selects a Raman-sideband pulse. Its joint molecule-motion propagator is

$$
U_a=\mathcal T\exp\!\left[-\frac{i}{\hbar}\int_0^{\tau_a}H_a(t)\,dt\right].
$$

The motional measurement is binary: $k=0$ for $n=0$ and $k=1$ for $n\geq1$. Precomputed branch matrices are

$$
(B_{a,k})_{ji}
=\sum_{n\in\mathcal N_k}
\left|\langle j,n|U_a|i,0\rangle\right|^2,
\qquad
\sum_{k,j}(B_{a,k})_{ji}=1.
$$

For population $s$:

$$
\widetilde s_{a,k}=B_{a,k}s,
\qquad
p_k(s,a)=\mathbf 1^T\widetilde s_{a,k},
\qquad
s_{a,k}^{\mathrm{cond}}=\frac{\widetilde s_{a,k}}{p_k(s,a)}.
$$

`RLQLSEnv.step(a)` samples $k\sim\mathrm{Categorical}(p_0,p_1)$. `branch_details(s,a)` returns both branches for qMDP learning.

### Blackbody-radiation propagation

For rates $R_{i\to j}$ and column populations,

$$
\dot s=G_{\mathrm{BBR}}s,
\qquad
G_{ji}=R_{i\to j}\;(j\neq i),
\qquad
G_{ii}=-\sum_{j\neq i}R_{i\to j}.
$$

The optional action-dependent noise step is

$$
s'_{a,k}=T_{\mathrm{BBR}}(\tau_a)s_{a,k}^{\mathrm{cond}},
\qquad
T_{\mathrm{BBR}}(\tau_a)=e^{G_{\mathrm{BBR}}\tau_a}.
$$

## Control objective

Preparation succeeds at the stopping time

$$
\tau_\eta=\inf\{t:\max_i s_{t,i}\geq1-\eta\}.
$$

With $r_t=-1$, maximizing return minimizes expected pulse count. The optional overlap penalty uses

$$
o(s,s')=\frac{s^Ts'}{\lVert s\rVert_2\lVert s'\rVert_2},
\qquad
r=-1-r_o\mathbf 1\!\left[o(s,s')>1-\frac1{N_S}\right].
$$

The qMDP Double-DQN target averages all measurement outcomes:

$$
y(s,a)=\sum_k p_k(s,a)\left[
r_k+\gamma(1-d_k)
Q_{\bar\theta}\!\left(s'_k,
\arg\max_{a'}Q_\theta(s'_k,a')\right)
\right].
$$

Here $Q_\theta$ selects the next action, $Q_{\bar\theta}$ evaluates it, and $d_k$ marks a terminal branch. The target network follows

$$
\bar\theta\leftarrow(1-\tau)\bar\theta+\tau\theta.
$$

## Workflow

$$
\{H_a(t)\}
\longrightarrow
\{B_{a,k}\}
\longrightarrow
\text{Gym trajectories}
\longrightarrow
Q_\theta(s,a)
\longrightarrow
\widehat{P}(\tau_\eta\leq n).
$$

Code locations:

- `src/rlqls/qutip_builder.py`: $H_a(t)\mapsto B_{a,k}$ through coherent QuTiP propagation.
- `src/rlqls/model.py`: branch maps, normalization, and Boltzmann state.
- `src/rlqls/materials.py`: CaH$^+$/H$_3$O$^+$ surrogate construction.
- `src/rlqls/bbr.py`: $G_{\mathrm{BBR}}$ and $T_{\mathrm{BBR}}$.
- `src/rlqls/env.py`: sampled measurement trajectory and complete branch API.
- `src/rlqls/dqn.py`: sampled/qMDP Double-DQN targets.
- `src/rlqls/evaluation.py`: greedy and pulse-sweeping Monte Carlo evaluation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test,physics]'
```

The `physics` extra installs QuTiP for exact propagation when complete Hamiltonians are available.

## Run

CaH$^+$:

```bash
python scripts/run_experiment.py \
  --material CaH \
  --episodes 1000 \
  --eval-episodes 3000 \
  --seed 7 \
  --output results/my_cah_run
```

H$_3$O$^+$:

```bash
python scripts/run_experiment.py \
  --material H3O \
  --episodes 100 \
  --eval-episodes 200 \
  --seed 41 \
  --batch-size 32 \
  --train-every 8 \
  --output results/my_h3o_run
```

Use `--update-mode qmdp` for the expected-branch target or `--update-mode sampled` for ordinary sampled Double DQN.

## Verification

```bash
PYTHONPATH=src pytest -q tests
```

Tests cover branch-map trace preservation, normalized outcome probabilities, Gymnasium returns, and CaH$^+$/H$_3$O$^+$ dimensions.

## Scope

The control/RL equations are implemented directly. Numerical reproduction is limited by unpublished pulse definitions and transition matrices: CaH$^+$ is a reduced-model reconstruction; H$_3$O$^+$ is a partial sensitivity model. See `results/reproduction_report.md` before interpreting numerical agreement.
