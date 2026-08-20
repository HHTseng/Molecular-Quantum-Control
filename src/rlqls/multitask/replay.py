"""Balanced replay for heterogeneous molecule tasks."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import random

import numpy as np


@dataclass(slots=True)
class MultiReplayItem:
    task_name: str
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    step_count: int
    next_step_count: int


class BalancedReplayBuffer:
    """Per-task bounded queues with approximately balanced minibatch sampling."""

    def __init__(self, capacity_per_task: int, seed: int) -> None:
        if capacity_per_task <= 0:
            raise ValueError("capacity_per_task must be positive")
        self.capacity_per_task = int(capacity_per_task)
        self.data: dict[str, deque[MultiReplayItem]] = defaultdict(
            lambda: deque(maxlen=self.capacity_per_task)
        )
        self.rng = random.Random(seed)

    def append(self, item: MultiReplayItem) -> None:
        self.data[item.task_name].append(item)

    def sample(
        self,
        batch_size: int,
        *,
        task_names: tuple[str, ...] | None = None,
    ) -> list[MultiReplayItem]:
        available = [
            name
            for name, queue in self.data.items()
            if queue and (task_names is None or name in task_names)
        ]
        if not available:
            raise ValueError("replay buffer is empty")
        selected: list[MultiReplayItem] = []
        base, remainder = divmod(batch_size, len(available))
        for index, name in enumerate(sorted(available)):
            count = base + (1 if index < remainder else 0)
            queue = self.data[name]
            if count <= len(queue):
                selected.extend(self.rng.sample(list(queue), count))
            else:
                selected.extend(self.rng.choices(list(queue), k=count))
        self.rng.shuffle(selected)
        return selected

    def __len__(self) -> int:
        return sum(len(queue) for queue in self.data.values())

    def count(self, task_name: str) -> int:
        return len(self.data.get(task_name, ()))


__all__ = ["MultiReplayItem", "BalancedReplayBuffer"]
