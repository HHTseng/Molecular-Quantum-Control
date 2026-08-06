#!/usr/bin/env python3
"""Check physicality ``sum_(k,j) B[a,k,j,i]=1`` for every surrogate map."""

from pathlib import Path
import sys, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rlqls import CaH16_surrogate, H3O_130_surrogate

for model in [CaH16_surrogate(), H3O_130_surrogate(ROOT / "data")]:
    # Maximum trace-preservation error over pulse actions a and source states i.
    error = np.max(np.abs(model.branch_matrices.sum(axis=(1, 2)) - 1))
    print(model.metadata["material"], model.n_states, model.n_actions, "trace error", error)
    print("  uncertainties:")
    for text in model.metadata["uncertainties"]:
        print("   -", text)
