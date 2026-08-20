"""Gymnasium compatibility layer.

The project targets Gymnasium 1.3.0.  The real package is used whenever it is
installed.  A deliberately small fallback is provided only for the network-
isolated execution environment used to validate this repository.

The multi-molecule environment uses ``spaces.Dict`` because its observation is
a padded graph record.  The fallback implements only the subset required by
our tests; production training should install Gymnasium.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

try:  # pragma: no cover - normal installation path.
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - sandbox fallback.
    GYMNASIUM_AVAILABLE = False

    class _Env:
        metadata: dict[str, Any] = {}

        def __init__(self) -> None:
            self.np_random = np.random.default_rng()

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ):
            del options
            if seed is not None:
                self.np_random = np.random.default_rng(seed)
            elif not hasattr(self, "np_random"):
                self.np_random = np.random.default_rng()
            return None

        def close(self) -> None:
            return None

    class _Discrete:
        def __init__(self, n: int) -> None:
            if n <= 0:
                raise ValueError("n must be positive")
            self.n = int(n)

        def contains(self, x: Any) -> bool:
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

        def sample(self, mask: np.ndarray | None = None) -> int:
            if mask is None:
                return int(np.random.randint(self.n))
            valid = np.flatnonzero(np.asarray(mask, dtype=np.int8) != 0)
            if valid.size == 0:
                raise ValueError("action mask has no valid action")
            return int(np.random.choice(valid))

    class _Box:
        def __init__(
            self,
            low: float | int | np.ndarray,
            high: float | int | np.ndarray,
            shape: tuple[int, ...] | None = None,
            dtype=np.float32,
        ) -> None:
            self.dtype = np.dtype(dtype)
            if shape is None:
                low_array = np.asarray(low, dtype=self.dtype)
                high_array = np.asarray(high, dtype=self.dtype)
                if low_array.shape != high_array.shape:
                    raise ValueError("low/high shape mismatch")
                self.shape = low_array.shape
                self.low = low_array
                self.high = high_array
            else:
                self.shape = tuple(shape)
                self.low = np.full(self.shape, low, dtype=self.dtype)
                self.high = np.full(self.shape, high, dtype=self.dtype)

        def contains(self, x: Any) -> bool:
            a = np.asarray(x)
            return (
                a.shape == self.shape
                and np.all(a >= self.low)
                and np.all(a <= self.high)
            )

    class _Dict:
        def __init__(self, mapping: Mapping[str, Any]) -> None:
            self.spaces = dict(mapping)

        def contains(self, x: Any) -> bool:
            if not isinstance(x, Mapping):
                return False
            if set(x) != set(self.spaces):
                return False
            return all(space.contains(x[key]) for key, space in self.spaces.items())

    gym = SimpleNamespace(Env=_Env)
    spaces = SimpleNamespace(Discrete=_Discrete, Box=_Box, Dict=_Dict)

__all__ = ["gym", "spaces", "GYMNASIUM_AVAILABLE"]
