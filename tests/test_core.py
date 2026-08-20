from pathlib import Path
import numpy as np
from rlqls import RLQLSEnv,build_cah16_surrogate,build_h3o130_surrogate

def test_cah_model_and_env():
    m=build_cah16_surrogate();assert m.branch_matrices.shape==(13,2,16,16)
    assert np.allclose(m.branch_matrices.sum(axis=(1,2)),1,atol=2e-5)
    e=RLQLSEnv(m,max_steps=10);s,info=e.reset(seed=1);assert s.shape==(16,)
    out=e.step(0);assert len(out)==5 and np.isclose(out[4]["branch_probabilities"].sum(),1)

def test_h3o_count():
    root=Path(__file__).resolve().parents[1]
    m=build_h3o130_surrogate(root/"data");assert m.branch_matrices.shape==(218,2,130,130)
