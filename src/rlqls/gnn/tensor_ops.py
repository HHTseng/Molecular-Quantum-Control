"""Masked/scatter tensor operations adapted from the size-aware QDX pattern."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        *,
        activation: type[nn.Module] = nn.SiLU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        widths = (input_dim,) + tuple(hidden_dims) + (output_dim,)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(left, right))
            if index + 1 < len(widths) - 1:
                if layer_norm:
                    layers.append(nn.LayerNorm(right))
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    numerator = torch.sum(values * weights, dim=dim)
    denominator = torch.sum(weights, dim=dim).clamp_min(1.0)
    return numerator / denominator


def masked_max(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    negative = torch.finfo(values.dtype).min
    masked = torch.where(mask.unsqueeze(-1), values, torch.full_like(values, negative))
    maximum = torch.amax(masked, dim=dim)
    any_valid = torch.any(mask, dim=dim)
    return torch.where(any_valid.unsqueeze(-1), maximum, torch.zeros_like(maximum))


def gather_nodes(nodes: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather node embeddings from ``nodes[B,N,H]`` at ``indices[B,E]``."""

    if nodes.ndim != 3 or indices.ndim != 2:
        raise ValueError("gather_nodes expects nodes[B,N,H] and indices[B,E]")
    index = indices.to(torch.int64).unsqueeze(-1).expand(-1, -1, nodes.shape[-1])
    return torch.gather(nodes, 1, index)


def aggregate_messages(
    messages: torch.Tensor,
    receivers: torch.Tensor,
    edge_mask: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Masked receiver-wise scatter mean for edge messages."""

    if messages.ndim != 3:
        raise ValueError("messages must have shape [B,E,H]")
    batch, _, hidden = messages.shape
    receiver = receivers.to(torch.int64)
    weights = edge_mask.to(messages.dtype)
    weighted = messages * weights.unsqueeze(-1)
    index = receiver.unsqueeze(-1).expand(-1, -1, hidden)
    summed = torch.zeros(
        (batch, num_nodes, hidden),
        dtype=messages.dtype,
        device=messages.device,
    ).scatter_add_(1, index, weighted)
    count = torch.zeros(
        (batch, num_nodes),
        dtype=messages.dtype,
        device=messages.device,
    ).scatter_add_(1, receiver, weights)
    return summed / count.unsqueeze(-1).clamp_min(1.0)


def aggregate_actions(
    records: torch.Tensor,
    action_indices: torch.Tensor,
    record_mask: torch.Tensor,
    num_actions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool a variable-cardinality transition set into mean/max per pulse.

    ``records[b,t]`` is the embedding of transition record ``t`` belonging to
    action ``action_indices[b,t]``.  This is the hyperedge-like generalization
    of QDX's one-/two-endpoint action scorer.
    """

    if records.ndim != 3:
        raise ValueError("records must have shape [B,T,H]")
    batch, _, hidden = records.shape
    action = action_indices.to(torch.int64)
    weights = record_mask.to(records.dtype)
    index = action.unsqueeze(-1).expand(-1, -1, hidden)

    summed = torch.zeros(
        (batch, num_actions, hidden),
        dtype=records.dtype,
        device=records.device,
    ).scatter_add_(1, index, records * weights.unsqueeze(-1))
    count = torch.zeros(
        (batch, num_actions),
        dtype=records.dtype,
        device=records.device,
    ).scatter_add_(1, action, weights)
    mean = summed / count.unsqueeze(-1).clamp_min(1.0)

    negative = torch.finfo(records.dtype).min
    for_max = torch.where(
        record_mask.unsqueeze(-1),
        records,
        torch.full_like(records, negative),
    )
    maximum = torch.full(
        (batch, num_actions, hidden),
        negative,
        dtype=records.dtype,
        device=records.device,
    )
    maximum.scatter_reduce_(1, index, for_max, reduce="amax", include_self=True)
    maximum = torch.where(
        (count > 0).unsqueeze(-1), maximum, torch.zeros_like(maximum)
    )
    return mean, maximum


__all__ = [
    "MLP",
    "masked_mean",
    "masked_max",
    "gather_nodes",
    "aggregate_messages",
    "aggregate_actions",
]
