"""Atom/isotope message passing and explicit chemistry-context fusion."""
from __future__ import annotations

import torch
from torch import nn

from rlqls.multitask.observation import MolecularGraphObservation

from .tensor_ops import MLP, aggregate_messages, gather_nodes, masked_max, masked_mean


class ChemistryEncoder(nn.Module):
    r"""Compute an episode-static chemistry context ``c_m``.

    For atom embeddings ``z_r`` and directed atom edges ``r -> q``:

    .. math::
        \mu_{rq}^{(\ell)} = M_\ell(z_r,z_q,e_{rq},c_{\rm explicit}),
        \quad
        z_q^{(\ell+1)} = U_\ell(z_q,\sum_r \mu_{rq},c_{\rm explicit}).
    """

    def __init__(
        self,
        atom_dim: int,
        atom_edge_dim: int,
        explicit_dim: int,
        hidden_dim: int,
        context_dim: int,
        *,
        num_layers: int = 2,
        use_atomistic: bool = True,
        use_explicit: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.use_atomistic = use_atomistic
        self.use_explicit = use_explicit

        self.atom_embed = MLP(atom_dim, (hidden_dim,), hidden_dim)
        self.edge_embed = MLP(atom_edge_dim, (hidden_dim,), hidden_dim)
        self.explicit_embed = MLP(2 * explicit_dim, (hidden_dim,), hidden_dim)
        self.message_mlps = nn.ModuleList(
            [MLP(4 * hidden_dim, (hidden_dim,), hidden_dim) for _ in range(num_layers)]
        )
        self.update_mlps = nn.ModuleList(
            [MLP(3 * hidden_dim, (hidden_dim,), hidden_dim) for _ in range(num_layers)]
        )
        self.readout = MLP(3 * hidden_dim, (hidden_dim,), context_dim)

    def forward(self, obs: MolecularGraphObservation) -> torch.Tensor:
        atom_input = obs.atom_features
        if not self.use_atomistic:
            atom_input = torch.zeros_like(atom_input)
        z = self.atom_embed(atom_input) * obs.atom_mask.unsqueeze(-1).to(atom_input.dtype)
        edge = self.edge_embed(obs.atom_edge_features)

        explicit_input = torch.cat(
            [
                obs.explicit_chemistry_features
                * obs.explicit_chemistry_feature_mask,
                obs.explicit_chemistry_feature_mask,
            ],
            dim=-1,
        )
        if not self.use_explicit:
            explicit_input = torch.zeros_like(explicit_input)
        explicit = self.explicit_embed(explicit_input)

        for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps):
            sender = gather_nodes(z, obs.atom_senders)
            receiver = gather_nodes(z, obs.atom_receivers)
            context = explicit.unsqueeze(1).expand(-1, sender.shape[1], -1)
            messages = message_mlp(torch.cat([sender, receiver, edge, context], dim=-1))
            aggregate = aggregate_messages(
                messages,
                obs.atom_receivers,
                obs.atom_edge_mask,
                z.shape[1],
            )
            context_nodes = explicit.unsqueeze(1).expand(-1, z.shape[1], -1)
            delta = update_mlp(torch.cat([z, aggregate, context_nodes], dim=-1))
            z = (z + delta) * obs.atom_mask.unsqueeze(-1).to(z.dtype)

        mean = masked_mean(z, obs.atom_mask, dim=1)
        maximum = masked_max(z, obs.atom_mask, dim=1)
        return self.readout(torch.cat([mean, maximum, explicit], dim=-1))


__all__ = ["ChemistryEncoder"]
