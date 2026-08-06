"""Plot learning curves and the stopping-time CDF used in paper comparisons."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .evaluation import EvaluationResult


def moving_average(values, window: int = 100):
    """Return the finite-window mean ``bar{x}_t=(1/w)sum x_j``."""
    x = np.asarray(values, dtype=float)
    if len(x) < window:
        return np.arange(1, len(x) + 1), x
    kernel = np.ones(window) / window
    return np.arange(window, len(x) + 1), np.convolve(x, kernel, mode="valid")


def plot_training(lengths, output: Path, window: int = 100, title: str = "Training episode length"):
    """Plot sampled episode stopping/censoring lengths and their moving mean."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    x, y = moving_average(lengths, window)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(lengths) + 1), lengths, alpha=0.25, label="episode")
    ax.plot(x, y, label=f"moving average ({window})")
    ax.set_xlabel("training episode")
    ax.set_ylabel("steps")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def completion_curve(result: EvaluationResult, max_steps: int):
    """Tabulate ``F(n)=P_hat(tau_eta<=n)`` for ``n=1,...,max_steps``."""
    steps = np.arange(1, max_steps + 1)
    fractions = np.asarray([result.completion_fraction(int(n)) for n in steps])
    return steps, fractions


def plot_completion(
    results: dict[str, EvaluationResult],
    output: Path,
    max_steps: int,
    title: str = "Completion probability",
):
    """Plot empirical preparation-completion CDFs for several policies."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, result in results.items():
        x, y = completion_curve(result, max_steps)
        ax.plot(x, y, label=label)
    ax.set_xlabel("number of pulse-measurement steps")
    ax.set_ylabel("fraction completed")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


__all__ = ["moving_average", "plot_training", "plot_completion"]
