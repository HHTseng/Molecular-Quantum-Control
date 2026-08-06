# RL-QLS implementation and reproduction report

## 1. Scope and conclusion

This implementation reproduces the **algorithmic structure** of RL-QLS and provides executable Gymnasium/PyTorch code for CaH+ and H3O+. The control layer follows the main paper and Supplemental Secs. SA--SD:

- RL state: molecular population vector $s_t\in\Delta_{N_S-1}$;
- RL action: one pulse index $a_t$ from a finite library;
- fixed environment operation: pulse, projective motional measurement, conditional state update, and motional reset;
- base reward: $-1$ per pulse-measurement step;
- optional physics-informed overlap penalty;
- sampled Double DQN and expected-branch qMDP Double DQN;
- experience replay, target network, soft target update, and epsilon-greedy exploration.

The **CaH+ reduced example is quantitatively close** in mean RL episode length. Across three qMDP seeds, the reconstructed mean is

$$
8.51\pm0.30\ \text{steps},
$$

compared with $8.3$ in the paper. The full H3O+ result is **not reproduced** because the exact 218-pulse library and its $218\times2$ branch matrices are not published; the transparent reconstruction reaches only 12% greedy completion in the more paper-faithful four-motional-state run, versus 80% within 62 pulses and a 93.4% plateau reported in the paper.

This gap is attributable primarily to missing physics inputs, not merely to the DQN implementation.

---

## 2. Implemented mathematical model

Let

$$
\mathcal H_{\mathrm{mol}}=\operatorname{span}\{|i\rangle:i=1,\ldots,N_S\}
$$

and let the inter-step state be the diagonal molecular population

$$
s=(s_1,\ldots,s_{N_S})^T,\qquad s_i\geq0,\qquad \mathbf1^Ts=1.
$$

For pulse $a\in\{0,\ldots,N_A-1\}$ and binary motional result $k\in\{0,1\}$, the code stores

$$
B_{a,k}\in\mathbb R_{\geq0}^{N_S\times N_S},
\qquad
(B_{a,k})_{ji}
=
\sum_{n\in\mathcal N_k}
|\langle j,n|U_a|i,0\rangle|^2.
$$

The branch probability and normalized conditional state are

$$
\widetilde s_{a,k}=B_{a,k}s,
$$

$$
p_k(s,a)=\mathbf1^T\widetilde s_{a,k},
$$

$$
F_{a,k}(s)=\frac{\widetilde s_{a,k}}{p_k(s,a)}.
$$

The implementation explicitly normalizes the branch. This is required by main-text Eq. (3) and Supplemental Eq. (S3); main-text Eq. (4b) either suppresses this factor or treats its matrix as an already conditioned map.

The terminal set is

$$
\mathcal G_\eta
=
\{s\in\Delta_{N_S-1}:\|s\|_\infty\geq1-\eta\}.
$$

The reported runs use $\eta=0.01$.

### qMDP Double-DQN target

For online network $Q_\theta$ and target network $Q_{\bar\theta}$, the implemented target is

$$
y(s,a)
=
\sum_{k=0}^1p_k(s,a)
\left[
 r_k+
 \gamma(1-d_k)
 Q_{\bar\theta}\!\left(
 F_{a,k}(s),
 \underset{a'}{\arg\max}\,Q_\theta(F_{a,k}(s),a')
 \right)
\right].
$$

This combines Supplemental Eq. (S18) with the paper's separate statement that double-Q networks are used. The paper does not print the combined formula, so this is a reasonable but flagged implementation choice.

---

## 3. Gymnasium interface

The environment follows the Gymnasium 1.3 API:

```python
observation, info = env.reset(seed=seed)
observation, reward, terminated, truncated, info = env.step(action)
```

- `observation`: $s_t\in\mathbb R^{N_S}$;
- `action`: pulse index $a_t\in\{0,\ldots,N_A-1\}$;
- `info["measurement_outcome"]`: sampled $k_t\in\{0,1\}$;
- `info["all_branch_states"]`: both $F_{a_t,k}(s_t)$;
- `info["branch_probabilities"]`: both $p_k(s_t,a_t)$.

The measurement is **not** an RL action. It is a fixed stochastic environment operation after the chosen pulse.

The project pins `gymnasium==1.3.0`. The execution sandbox could not install packages from the internet, so the tests here used the included minimal API fallback; the public package code imports real Gymnasium whenever it is installed.

---

## 4. Physics preprocessing

### Exact path supported by the code

`qutip_builder.py` implements the paper's intended preprocessing:

1. supply the complete time-dependent Hamiltonian $H_a(t)$;
2. solve $U_a|i,0\rangle$ for every action $a$ and input state $i$;
3. retain coherence during propagation;
4. aggregate $|\langle j,n|U_a|i,0\rangle|^2$ into $k=0$ and $k\geq1$;
5. verify
   $$
   \sum_{k,j}(B_{a,k})_{ji}=1.
   $$

The generic builder targets QuTiP 5.x. The paper itself used QuTiP 4.7, `sesolve`/`propagator`, and the tolerances reported in Supplemental Sec. SC.

### Why reconstructed models are needed here

The article does not provide:

- the full numerical CaH+ branch matrices;
- the exact H3O+ 218 pulse definitions;
- the H3O+ $218\times2$ matrices of size $130\times130$;
- the full H3O+ inversion-rotation Hamiltonian and coupling construction, which the paper defers to a subsequent article.

Therefore the repository includes explicit surrogate builders rather than pretending these quantities are known.

---

## 5. CaH+ reconstruction

### 5.1 Model

The reduced model uses:

- $N_S=16$ states, ordered I--XVI as in Supplemental Fig. S5;
- $N_A=13$ pulses;
- dominant blue-to-red transitions from Supplemental Fig. S2;
- pulse frequencies and durations from Supplemental Table S2;
- a 300 K Boltzmann prior;
- omitted weak yellow/off-resonant channels;
- inferred secondary transfer efficiencies for pulses 3, 4, and 9.

The last two points make the surrogate easier to purify than the paper's QuTiP model, especially for rare $k=1$ outcomes.

### 5.2 Training

Primary qMDP settings:

| Quantity | Value |
|---|---:|
| episodes | 1000 |
| hidden layers | $128,128,128$ |
| learning rate | $5\times10^{-4}$ |
| soft target rate | $10^{-3}$ |
| discount | $1$ |
| final epsilon | $0.005$ |
| replay batch | 64 |
| optimizer cadence | every 4 environment steps |
| maximum episode length | 100 |

The optimizer cadence and batch size are not specified by the paper; they are computational choices.

### 5.3 Results

Three independently trained qMDP models gave greedy mean episode lengths

$$
8.251,\quad 8.842,\quad 8.428,
$$

hence

$$
\boxed{8.51\pm0.30\ \text{steps}}.
$$

The paper reports $8.3$ steps. The relative difference of the three-seed mean is approximately $+2.5\%$.

| Metric | Paper | Reconstruction |
|---|---:|---:|
| mean RL episode length | 8.3 | $8.51\pm0.30$ |
| mean sweeping episode length | 9.7 | 8.79 |
| RL completion by 8 pulses | 56% | $53.7\%\pm2.6\%$ |
| RL completion by 18 pulses | 99% | $97.76\%\pm0.14\%$ |

The mean episode length and late completion fractions are close. The early-time distribution is not:

| Pulses | Paper RL | Reconstructed RL, 3-seed mean |
|---:|---:|---:|
| 2 | 0% | 9.8% |
| 3 | 15% | 20.8% |
| 4 | 35% | 26.9% |
| 5 | 35% | 31.2% |
| 6 | 35% | 38.7% |
| 7 | 45% | 48.6% |
| 8 | 56% | 53.7% |
| 18 | 99% | 97.8% |

The nonzero two-pulse success is a known surrogate artifact: a dominant-edge-only pulse can make the excited-motion branch point to a unique molecular destination, while the full published figure shows additional weak channels.

The reconstructed sweeping mean is 9.4% below the paper's 9.7, confirming that the surrogate is too easy. Consequently, the RL advantage over sweeping is smaller than in the article.

---

## 6. H3O+ reconstruction

### 6.1 Data extracted exactly from the supplement

The repository contains:

- 130 state energies from Supplemental Table S3;
- 371 Raman rates from Supplemental Table S4;
- the paper's cutoff $\Omega/(2\pi)\gtrsim0.1\,\mathrm{kHz}$.

Filtering and adjacent-frequency clustering with tolerance $0.43\,\mathrm{kHz}$ produces exactly 218 groups. The tolerance is **inferred solely to reproduce the reported action count**; it is not given by the paper.

### 6.2 More paper-faithful surrogate

The default H3O+ builder:

- retains four motional Fock states, as stated for Fig. S10;
- solves a local coherent rotating-frame Hamiltonian;
- aggregates $n=0$ into $k=0$ and $n=1,2,3$ into $k=1$;
- uses the strongest transition in each inferred frequency cluster as the pulse center and ideal $\pi$-time reference;
- includes tabulated transitions within a $2\,\mathrm{kHz}$ local window.

The strongest-transition rule, cluster rule, and local window are reconstruction assumptions.

### 6.3 Partial training result

A 100-episode qMDP run, using the article's H3O+-reported values $r_o=1$, $\gamma=0.9$, learning rate $5\times10^{-4}$, target rate $10^{-4}$, and $\epsilon_{\mathrm{end}}=0.125$, produced:

| Metric | Paper | Four-motion reconstruction |
|---|---:|---:|
| completion by 62 pulses | 80% | 12% |
| reported plateau | 93.4% | 12% through 150 pulses |
| sweeping by 150 pulses | not given as one scalar | 36% |

The successful reconstructed RL episodes terminate in about two steps, while the remaining episodes generally hit the time limit. This is not the qualitative behavior of the paper. It indicates that the inferred pulse model contains a few lottery-like nearly pure branches but does not supply the connected, informative action graph learned in the authors' exact model.

A less faithful binary-motion/cluster-median sensitivity model reached 38% completion after 300 episodes. That larger number is not used as the primary result because the model omits the four-motional-state propagation explicitly reported in the supplement.

### 6.4 Why exact H3O+ reproduction is currently blocked

The H3O+ result is highly sensitive to:

1. which nominally degenerate transitions are assigned to one pulse;
2. pulse center frequency and duration;
3. weak off-resonant couplings;
4. polarization/direction conventions;
5. coherent interference among multilevel paths;
6. higher motional states;
7. the exact reward and training cadence.

Tables S3 and S4 do not uniquely determine these choices. Fig. S10 is a rasterized visualization, not a numerical matrix release. Thus many incompatible 218-action environments are consistent with the public supplement.

---

## 7. Uncertainties and implementation choices

### Directly supported by the paper

- population-vector RL state;
- discrete pulse action;
- projective motional measurement after every pulse;
- branch probabilities from pulse-conditioned dynamics;
- $-1$ reward per step;
- terminal confidence $\max_i s_i\geq1-\eta$;
- DQN with a fully connected 128-node architecture;
- replay, double-Q networks, epsilon-greedy exploration, soft target updates;
- qMDP expectation over both measurement outcomes;
- H3O+ hyperparameters $\gamma=0.9$, $r_o=1$, learning rate $5\times10^{-4}$, target rate $10^{-4}$, and final epsilon $0.125$ for the reported run.

### Best assumptions, explicitly flagged in code

- three **hidden** layers of 128 units; the phrase "three-layer" is ambiguous;
- Double-DQN selection/evaluation inside each qMDP branch;
- branch-specific overlap reward inside the qMDP expectation;
- additive overlap penalty
  $$
  r=-1-r_o\mathbf1[o(s,s')>1-1/N_S];
  $$
- corrected epsilon schedule
  $$
  \epsilon(n)=\epsilon_{\rm end}+(\epsilon_{\rm start}-\epsilon_{\rm end})e^{-n/\tau_\epsilon};
  $$
  the printed Supplemental Eq. (S16) has the opposite sign and does not start at $\epsilon_{\rm start}$;
- replay batch size and update cadence;
- H3O+ pulse clustering, center, duration, and detuning window;
- sequential BBR application after measurement conditioning.

### Not reproduced

- CaH+ BBR curves, because numerical Einstein-rate inputs were not released in machine-readable form;
- exact H3O+ pulse library;
- exact H3O+ success curve;
- qMDP-vs-MDP loss-magnitude plot;
- full paper hyperparameter grid searches.

---

## 8. Software verification

The executed test suite reports:

```text
2 passed
```

It checks:

- CaH+ shape $(13,2,16,16)$;
- H3O+ shape $(218,2,130,130)$;
- trace preservation of every input column;
- normalized branch probabilities;
- Gymnasium-compatible `reset` and `step` return values.

The largest observed H3O+ trace-preservation error in the four-motion model is approximately $1.2\times10^{-7}$ before the `BranchModel` repair/normalization step.

---

## 9. Related public code used as structural guidance

No official repository for this paper was located. The closest useful public resources were:

- the official Gymnasium custom-environment tutorial;
- the official PyTorch DQN tutorial, for replay memory, policy/target networks, and soft updates;
- `pschindler/qutip-molecule-showcase`, for physically related molecular quantum-logic spectroscopy simulations in QuTiP;
- current QuTiP solver documentation.

The GitHub repository named `RL_QLS` found in search is unrelated; it concerns quantum local search for combinatorial optimization.

---

## 10. Artifact map

- `README.md`: installation and usage.
- `PSEUDOCODE_REFERENCE.md`: detailed algorithm pseudocode.
- `src/rlqls/`: implementation.
- `scripts/run_experiment.py`: configurable training/evaluation CLI.
- `scripts/generate_figures.py`: regeneration of report figures.
- `tests/`: smoke and invariant tests.
- `results/checkpoints/`: trained PyTorch checkpoints.
- `results/figures/cah_completion.png`: CaH+ paper/reconstruction comparison.
- `results/figures/h3o_completion.png`: H3O+ partial comparison.
- `results/*.json`: raw histories and metrics.

## 11. Bottom line

The implementation is suitable for:

- testing RL/control logic;
- replacing surrogate $B_{a,k}$ matrices with exact QuTiP or experimental maps;
- studying qMDP versus sampled updates;
- extending actions/rewards/noise models;
- preparing a direct reproduction once the authors' numerical pulse library is available.

The CaH+ result is a useful reduced-model reproduction. The H3O+ result should be treated as a diagnostic of missing model information, not as a faithful benchmark of the published method.
