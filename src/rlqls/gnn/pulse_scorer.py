"""Variable-cardinality candidate-pulse scoring head."""
from __future__ import annotations

import torch
from torch import nn

from rlqls.multitask.observation import MolecularGraphObservation

from .tensor_ops import MLP, aggregate_actions, gather_nodes


class PulseScorer(nn.Module):
    r"""Score every valid pulse from the transitions it affects.

    A pulse action ``a`` is treated as a hyperedge-like set

    .. math:: \mathcal T_a=\{(i,j,z_{aij})\}.

    Each record is embedded from source/destination level embeddings, its
    physical/branch descriptor, the pulse descriptor, and global context.  Mean
    and max pooling over ``T_a`` gives an invariant pulse representation.
    """

    def __init__(
        self,
        level_hidden_dim: int,
        pulse_dim: int,
        transition_dim: int,
        context_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.pulse_embed = MLP(2 * pulse_dim, (hidden_dim,), hidden_dim)
        self.transition_embed = MLP(transition_dim, (hidden_dim,), hidden_dim)
        self.record_mlp = MLP(
            5 * hidden_dim,
            (hidden_dim, hidden_dim),
            hidden_dim,
        )
        self.score_mlp = MLP(
            4 * hidden_dim,
            (hidden_dim, hidden_dim),
            1,
        )
        self.level_project = (
            nn.Identity()
            if level_hidden_dim == hidden_dim
            else nn.Linear(level_hidden_dim, hidden_dim)
        )
        self.context_project = (
            nn.Identity()
            if context_dim == hidden_dim
            else nn.Linear(context_dim, hidden_dim)
        )

    def forward(
        self,
        obs: MolecularGraphObservation,
        level_embeddings: torch.Tensor,
        control_context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pulse_input = torch.cat(
            [obs.pulse_features * obs.pulse_feature_mask, obs.pulse_feature_mask],
            dim=-1,
        )
        pulse_h = self.pulse_embed(pulse_input)
        level_h = self.level_project(level_embeddings)
        context_h = self.context_project(control_context)

        source = gather_nodes(level_h, obs.pulse_transition_src)
        destination = gather_nodes(level_h, obs.pulse_transition_dst)
        transition_h = self.transition_embed(obs.pulse_transition_features)
        action_index = obs.pulse_transition_action.to(torch.int64)
        action_gather = action_index.unsqueeze(-1).expand(-1, -1, pulse_h.shape[-1])
        pulse_for_record = torch.gather(pulse_h, 1, action_gather)
        context_for_record = context_h.unsqueeze(1).expand(-1, source.shape[1], -1)
        record_h = self.record_mlp(
            torch.cat(
                [
                    source,
                    destination,
                    transition_h,
                    pulse_for_record,
                    context_for_record,
                ],
                dim=-1,
            )
        )
        mean_pool, max_pool = aggregate_actions(
            record_h,
            action_index,
            obs.pulse_transition_mask,
            pulse_h.shape[1],
        )
        context_actions = context_h.unsqueeze(1).expand(-1, pulse_h.shape[1], -1)
        q_values = self.score_mlp(
            torch.cat([mean_pool, max_pool, pulse_h, context_actions], dim=-1)
        ).squeeze(-1)
        q_values = torch.where(
            obs.action_mask,
            q_values,
            torch.full_like(q_values, -1.0e9),
        )
        return q_values, {
            "pulse_embeddings": torch.cat([mean_pool, max_pool, pulse_h], dim=-1),
            "record_embeddings": record_h,
        }


__all__ = ["PulseScorer"]
