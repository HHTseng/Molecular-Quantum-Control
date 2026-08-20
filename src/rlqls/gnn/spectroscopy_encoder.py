"""Chemistry-conditioned message passing over molecular internal levels."""
from __future__ import annotations

import torch
from torch import nn

from rlqls.multitask.observation import MolecularGraphObservation

from .tensor_ops import MLP, aggregate_messages, gather_nodes, masked_max, masked_mean


class SpectroscopyEncoder(nn.Module):
    def __init__(
        self,
        level_dim: int,
        level_edge_dim: int,
        global_dim: int,
        chemistry_context_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.level_embed = MLP(2 * level_dim, (hidden_dim,), hidden_dim)
        self.edge_embed = MLP(level_edge_dim, (hidden_dim,), hidden_dim)
        self.global_embed = MLP(global_dim, (hidden_dim,), hidden_dim)
        self.chemistry_project = MLP(
            chemistry_context_dim, (hidden_dim,), hidden_dim
        )
        self.message_mlps = nn.ModuleList(
            [MLP(5 * hidden_dim, (hidden_dim,), hidden_dim) for _ in range(num_layers)]
        )
        self.update_mlps = nn.ModuleList(
            [MLP(4 * hidden_dim, (hidden_dim,), hidden_dim) for _ in range(num_layers)]
        )
        self.global_readout = MLP(4 * hidden_dim, (hidden_dim,), hidden_dim)

    def forward(
        self,
        obs: MolecularGraphObservation,
        chemistry_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        level_input = torch.cat(
            [obs.level_features * obs.level_feature_mask, obs.level_feature_mask],
            dim=-1,
        )
        h = self.level_embed(level_input)
        h = h * obs.level_mask.unsqueeze(-1).to(h.dtype)
        edge = self.edge_embed(obs.level_edge_features)
        global_h = self.global_embed(obs.global_features)
        chemistry_h = self.chemistry_project(chemistry_context)

        for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps):
            sender = gather_nodes(h, obs.level_senders)
            receiver = gather_nodes(h, obs.level_receivers)
            global_edges = global_h.unsqueeze(1).expand(-1, sender.shape[1], -1)
            chemistry_edges = chemistry_h.unsqueeze(1).expand(-1, sender.shape[1], -1)
            messages = message_mlp(
                torch.cat(
                    [sender, receiver, edge, global_edges, chemistry_edges], dim=-1
                )
            )
            aggregate = aggregate_messages(
                messages,
                obs.level_receivers,
                obs.level_edge_mask,
                h.shape[1],
            )
            global_nodes = global_h.unsqueeze(1).expand(-1, h.shape[1], -1)
            chemistry_nodes = chemistry_h.unsqueeze(1).expand(-1, h.shape[1], -1)
            delta = update_mlp(
                torch.cat([h, aggregate, global_nodes, chemistry_nodes], dim=-1)
            )
            h = (h + delta) * obs.level_mask.unsqueeze(-1).to(h.dtype)

        mean = masked_mean(h, obs.level_mask, dim=1)
        maximum = masked_max(h, obs.level_mask, dim=1)
        control_context = self.global_readout(
            torch.cat([mean, maximum, global_h, chemistry_h], dim=-1)
        )
        return h, control_context


__all__ = ["SpectroscopyEncoder"]
