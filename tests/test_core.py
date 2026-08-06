from pathlib import Path
import numpy as np
from rlqls import RLQLSEnv, CaH16_surrogate, H3O_130_surrogate


def test_cah_model_and_env():
    """Check B-map trace preservation and normalized binary branch masses."""
    m = CaH16_surrogate()
    assert m.branch_matrices.shape == (13, 2, 16, 16)
    assert np.allclose(m.branch_matrices.sum(axis=(1, 2)), 1, atol=2e-5)
    e = RLQLSEnv(m, max_steps=10)
    s, info = e.reset(seed=1)
    assert s.shape == (16,)
    out = e.step(0)
    assert len(out) == 5 and np.isclose(out[4]["branch_probabilities"].sum(), 1)


def test_h3o_count():
    """Check the inferred paper dimensions ``N_A=218`` and ``N_S=130``."""
    root = Path(__file__).resolve().parents[1]
    m = H3O_130_surrogate(root / "data")
    assert m.branch_matrices.shape == (218, 2, 130, 130)
"""Core physicality and Gym-interface checks (pseudocode Sec. 19)."""
