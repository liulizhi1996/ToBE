import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj

from src.model.gcn import GCN
from src.model.transformer import Transformer


class GPSConv(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        pe_dim: int,
        dropout: float = 0.0,
        act: str = 'silu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
        flow: str = 'source_to_target'
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.pe_dim = pe_dim
        self.dropout = dropout

        assert flow in ['source_to_target', 'target_to_source']
        self.flow = flow
        self.conv = GCN(flow=flow)

        self.attn = Transformer(feat_dim, pe_dim)
        self.mlp = Sequential(
            Linear(feat_dim, feat_dim),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, feat_dim, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, feat_dim, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, feat_dim, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn.reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
        self,
        x_src: Tensor,
        x_dst: Tensor,
        pe_src: Tensor,
        pe_dst: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x_src, x_dst, edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.flow == 'source_to_target':
                h = h + x_dst
            else:   # self.flow == target_to_source
                h = h + x_src
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        # Global attention transformer-style model.
        if self.flow == 'source_to_target':
            h = self.attn(pe_dst, pe_src, x_dst, x_src)
        else:   # self.flow == target_to_source
            h = self.attn(pe_src, pe_dst, x_src, x_dst)

        h = F.dropout(h, p=self.dropout, training=self.training)
        # Residual connection.
        if self.flow == 'source_to_target':
            h = h + x_dst
        else:  # self.flow == target_to_source
            h = h + x_src
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.feat_dim}, '
                f'{self.pe_dim}, flow={self.flow}')
