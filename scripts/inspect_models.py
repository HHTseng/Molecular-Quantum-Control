#!/usr/bin/env python3
from pathlib import Path
import sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from rlqls import build_cah16_surrogate,build_h3o130_surrogate
for model in [build_cah16_surrogate(),build_h3o130_surrogate(ROOT/"data")]:
    error=np.max(np.abs(model.branch_matrices.sum(axis=(1,2))-1))
    print(model.metadata["material"],model.n_states,model.n_actions,"trace error",error)
    print("  uncertainties:")
    for text in model.metadata["uncertainties"]:print("   -",text)
