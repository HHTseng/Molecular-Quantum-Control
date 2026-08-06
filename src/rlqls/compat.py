"""Gymnasium compatibility layer.

The package targets Gymnasium 1.3.0.  A tiny fallback is included only so this
network-isolated execution environment can validate the physics/RL code when
the external package cannot be installed.  Install Gymnasium for real use.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

try:  # pragma: no cover - used in a normal installation.
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - used only in this sandbox.
    GYMNASIUM_AVAILABLE = False

    class _Env:
        """Minimal API shim; physics remains implemented by ``RLQLSEnv``."""
        metadata: dict[str, Any] = {}

        def __init__(self) -> None:
            self.np_random = np.random.default_rng()

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            if seed is not None:
                self.np_random = np.random.default_rng(seed)
            elif not hasattr(self, "np_random"):
                self.np_random = np.random.default_rng()
            return None

        def close(self) -> None:
            return None

    class _Discrete:
        """Fallback representation of the finite pulse set ``A={0,...,N_A-1}``."""
        def __init__(self, n: int) -> None:
            if n <= 0:
                raise ValueError("n must be positive")
            self.n = int(n)

        def contains(self, x: Any) -> bool:
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class _Box:
        """Fallback bounding box containing the population simplex ``Delta``."""
        def __init__(
            self, low: float, high: float, shape: tuple[int, ...], dtype=np.float32
        ) -> None:
            self.low = float(low)
            self.high = float(high)
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)

        def contains(self, x: Any) -> bool:
            a = np.asarray(x)
            return a.shape == self.shape and np.all(a >= self.low) and np.all(a <= self.high)

    gym = SimpleNamespace(Env=_Env)
    spaces = SimpleNamespace(Discrete=_Discrete, Box=_Box)

__all__ = ["gym", "spaces", "GYMNASIUM_AVAILABLE"]
