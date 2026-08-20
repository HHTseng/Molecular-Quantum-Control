# Multi-Molecule GNN RL-QLS Transfer Validation

## Scope and status

This study validates the **integrated software architecture**: one shared
chemistry-conditioned GNN, molecule-specific state/action spaces, masked
candidate-pulse scoring, heterogeneous replay, and the expected-branch qMDP
Bellman update.  CaH$^+$ and H$_3$O$^+$ use
the existing reconstruction models.  MgH$^+$ and
D$_3$O$^+$ are controlled related-task deformations used only
to test transfer mechanics; they are not independent spectroscopy predictions.

For molecule $m$, pulse $a$, and motional outcome $k\in\{0,1\}$,

$$
q_k^{(m)}=B_{a,k}^{(m)}s,
\qquad
p_m(k\mid s,a)=\mathbf 1^\mathsf Tq_k^{(m)},
\qquad
F_{a,k}^{(m)}(s)=\frac{q_k^{(m)}}{p_m(k\mid s,a)}.
$$

The shared network scores only the local candidate set,

$$
a_t=\arg\max_{a\in\mathcal A_m}Q_\Theta(o_t^{(m)},a),
$$

and the heterogeneous Double-DQN qMDP target is

$$
y_b=\sum_{k=0}^1p_{b,k}
\left[r_{b,k}+\gamma(1-d_{b,k})
Q_{\bar\Theta}\!\left(o'_{b,k},
\arg\max_{a'\in\mathcal A_{m_b}}Q_\Theta(o'_{b,k},a')\right)\right].
$$

## Training protocol

- Source pretraining: 240 episodes on the round-robin schedule
  `(CaH+, CaH+, CaH+, H3O+)`, giving 180 CaH$^+$ and 60
  H$_3$O$^+$ episodes.
- Held-out target adaptation: 60 episodes per target for both fine-tuning and
  training from scratch.
- Joint reference: 160 episodes, 40 episodes per task.
- Evaluation: 500 independent Monte Carlo episodes per policy/task;
  brackets below are Wilson 95% intervals for the success probability.

## Source-task check

- **CaH+:** success 100.0%, mean censored length 9.22.
- **H3O+:** success 71.4%, mean censored length 11.62.

## Held-out related-task results

| Target | Policy | Success rate [95% CI] | Mean censored steps | Mean steps among successes |
|---|---|---:|---:|---:|
| MgH+ | zero-shot | 58.8% [54.4, 63.0] | 15.38 | 8.65 |
| MgH+ | zero-shot-no-chemistry | 49.6% [45.2, 54.0] | 15.95 | 6.75 |
| MgH+ | fine-tuned | 96.2% [94.1, 97.6] | 9.91 | 9.32 |
| MgH+ | scratch | 0.0% [0.0, 0.8] | 25.00 | n/a |
| MgH+ | joint | 18.0% [14.9, 21.6] | 21.55 | 5.84 |
| MgH+ | sweeping | 93.8% [91.3, 95.6] | 9.38 | 8.35 |
| MgH+ | random | 67.0% [62.8, 71.0] | 15.33 | 10.57 |
| D3O+ | zero-shot | 35.4% [31.3, 39.7] | 15.31 | 6.75 |
| D3O+ | zero-shot-no-chemistry | 43.8% [39.5, 48.2] | 14.65 | 7.78 |
| D3O+ | fine-tuned | 38.4% [34.2, 42.7] | 14.71 | 6.23 |
| D3O+ | scratch | 23.6% [20.1, 27.5] | 16.03 | 3.19 |
| D3O+ | joint | 19.6% [16.4, 23.3] | 16.73 | 3.33 |
| D3O+ | sweeping | 0.4% [0.1, 1.4] | 19.97 | 12.00 |
| D3O+ | random | 9.2% [7.0, 12.1] | 19.15 | 10.76 |

## Main observations

1. **MgH$^+$:** source pretraining transfers zero-shot
   (58.8%); 60 target episodes
   raise success to 96.2%,
   whereas the equal-budget scratch run remains at
   0.0%.  This is the clearest
   positive transfer result in the demonstration.
2. **D$_3$O$^+$:** zero-shot success is
   35.4%; fine-tuning reaches
   38.4%, compared with
   23.6% from scratch.  The gain
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
  $B_{a,k}$.  These are effective for validating variable-action scoring but
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
