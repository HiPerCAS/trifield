"""Mesh-native GATv2 surrogate.

    node features (N, 17) --encoder MLP--> (N, 128)
        --> 8 x [edge-conditioned GATv2 (4 heads) + residual + LayerNorm + LeakyReLU]
        --> decoder MLP 128 -> 64 -> 3  =  (psi, Phi_n, Phi_p) at every node

There is no global pooling: the same weights apply to a mesh of any size.
"""
from __future__ import annotations

import inspect

import torch

import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation


def make_act(name: str) -> nn.Module:
    return {"leaky": lambda: nn.LeakyReLU(negative_slope=0.01), "silu": nn.SiLU,
            "gelu": nn.GELU}[name]()


def make_norm(name: str, dim: int) -> nn.Module:
    return nn.LayerNorm(dim) if name == "layer" else nn.Identity()


class GATv2Block(nn.Module):
    """One edge-conditioned GATv2 message-passing layer with residual + LayerNorm."""

    def __init__(self, hidden_dim: int, heads: int = 4, edge_dim: int = 5,
                 dropout: float = 0.0, act: str = "leaky", norm: str = "layer"):
        super().__init__()
        self.conv = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False,
                              edge_dim=edge_dim, dropout=dropout)
        self.norm = make_norm(norm, hidden_dim)
        self.act = make_act(act)

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index, edge_attr=edge_attr)
        return self.act(self.norm(out + x))


class GlobalContext(nn.Module):
    """Learned virtual node: attention-pooled device summary broadcast back to every
    node (residual + LayerNorm). One read/write gives O(1) communication across the
    whole mesh, which the long-range (elliptic) electrostatic potential needs and an
    8-hop GNN cannot provide. Initialised as a no-op."""

    def __init__(self, hidden_dim: int, act_name: str = "leaky", norm: str = "layer"):
        super().__init__()
        act = make_act(act_name)
        self.pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act, nn.Linear(hidden_dim, 1)),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act, nn.Linear(hidden_dim, hidden_dim)))
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = make_norm(norm, hidden_dim)

    def forward(self, h, batch):
        if batch is None:
            batch = h.new_zeros(h.size(0), dtype=torch.long)
        g = self.proj(self.pool(h, index=batch))
        return self.norm(h + g[batch])


class DeviceGNN(nn.Module):
    """GATv2 network predicting per-node (psi, Phi_n, Phi_p) in volts.

    Parameters
    ----------
    in_dim : node feature dimension (17, see ``trifield.data.graph_builder``)
    hidden_dim : width of the message-passing layers (128)
    out_dim : number of predicted fields (3)
    n_layers : number of GATv2 blocks (8)
    heads : attention heads per block (4)
    edge_dim : edge feature dimension (5)
    dropout : attention dropout (0)
    grad_checkpoint : recompute each block in the backward pass instead of
        storing its activations (training only; identical gradients, ~8x less
        activation memory, slower epochs). Adds no parameters.
    """

    def __init__(self, in_dim: int = 17, hidden_dim: int = 128, out_dim: int = 3,
                 n_layers: int = 8, heads: int = 4, edge_dim: int = 5,
                 dropout: float = 0.0, grad_checkpoint: bool = False, global_node: bool = False,
                 act: str = "leaky", norm: str = "layer", input_norm: bool = False,
                 local_y: bool = False, drop_cols: tuple = ()):
        super().__init__()
        self.config = dict(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                           n_layers=n_layers, heads=heads, edge_dim=edge_dim, global_node=global_node,
                           act=act, norm=norm, input_norm=input_norm, local_y=local_y,
                           drop_cols=list(drop_cols))
        self.global_node = global_node
        self.local_y = local_y
        self.drop_cols = list(drop_cols)
        self.input_norm = input_norm
        if input_norm:
            # feature standardisation lives in the model so checkpoints are self-contained;
            # set_input_stats() fills these from the training set
            self.register_buffer("x_mean", torch.zeros(in_dim))
            self.register_buffer("x_std", torch.ones(in_dim))
            self.register_buffer("e_mean", torch.zeros(edge_dim))
            self.register_buffer("e_std", torch.ones(edge_dim))
        if global_node:
            self.global_blocks = nn.ModuleList([GlobalContext(hidden_dim, act, norm) for _ in range(n_layers)])
        self.grad_checkpoint = grad_checkpoint
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            make_act(act),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            GATv2Block(hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout, act=act, norm=norm)
            for _ in range(n_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            make_act(act),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    @classmethod
    def from_config(cls, config: dict, **overrides) -> "DeviceGNN":
        """Build a model from a checkpoint ``config`` dict (unknown keys are ignored)."""
        allowed = set(inspect.signature(cls.__init__).parameters) - {"self"}
        kwargs = {k: v for k, v in config.items() if k in allowed}
        kwargs.update(overrides)
        return cls(**kwargs)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """x: (N, in_dim), edge_index: (2, E), edge_attr: (E, edge_dim) -> (N, out_dim).
        ``batch`` (N,) assigns nodes to graphs (only used by the global node)."""
        x, edge_attr = self.prepare_inputs(x, edge_attr)
        h = self.encoder(x)
        for i, layer in enumerate(self.layers):
            if self.grad_checkpoint and self.training and h.requires_grad:
                h = checkpoint(layer, h, edge_index, edge_attr, use_reentrant=False)
            else:
                h = layer(h, edge_index, edge_attr)
            if self.global_node:
                h = self.global_blocks[i](h, batch)
        return self.decoder(h)

    def transform_features(self, x):
        """Feature engineering applied before standardisation (shared by stats and forward)."""
        if self.local_y:                          # position inside the own fin pitch, not fin index
            x = x.clone()
            x[:, 1] = x[:, 1] - torch.floor(x[:, 1])
        return x

    def prepare_inputs(self, x, edge_attr):
        x = self.transform_features(x)
        if self.input_norm:
            x = (x - self.x_mean) / self.x_std
            edge_attr = (edge_attr - self.e_mean) / self.e_std
        if self.drop_cols:
            x = x.clone()
            x[:, self.drop_cols] = 0.0
        return x, edge_attr

    @torch.no_grad()
    def set_input_stats(self, ds, max_graphs: int = 400) -> None:
        """Per-column mean / std of node and edge features over (a sample of) a dataset.
        One-hot / flag columns keep std = 1 so they stay 0/1-valued."""
        if not self.input_norm:
            return
        idx = range(0, len(ds), max(1, len(ds) // max_graphs))
        xs, es = [], []
        for i in idx:
            d = ds[i]
            xs.append(self.transform_features(d.x.to(self.x_mean.device)))
            es.append(d.edge_attr.to(self.e_mean.device))
        X, E = torch.cat(xs), torch.cat(es)
        xm, xsd = X.mean(0), X.std(0).clamp(min=1e-6)
        binary = ((X == 0) | (X == 1)).all(0)
        xm[binary], xsd[binary] = 0.0, 1.0
        em, esd = E.mean(0), E.std(0).clamp(min=1e-6)
        eb = ((E == 0) | (E == 1)).all(0)
        em[eb], esd[eb] = 0.0, 1.0
        self.x_mean.copy_(xm); self.x_std.copy_(xsd); self.e_mean.copy_(em); self.e_std.copy_(esd)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


TriFieldGNN = DeviceGNN
