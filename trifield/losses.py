"""Training losses.

``DataLoss``: per-node MSE in z-score space over interior (non-contact) nodes.
``ContinuityResidualLoss``: finite-volume steady-state current continuity,
    div(J_n) = div(J_p) = 0, evaluated on the mesh edges in physical units.
``PoissonResidualLoss``: optional electrostatic Poisson residual.
``CombinedLoss``: L = w_data L_data + lambda(t) (w_cont L_cont + w_pois L_pois).

The physics terms are soft regularisers at the discretisation level of the
solver (edge lengths, edge-averaged Boltzmann densities, constant mobility),
not a re-implementation of the Scharfetter-Gummel scheme.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

# Physical constants (SI unless stated)
Q = 1.602176634e-19        # elementary charge [C]
K_B = 1.380649e-23         # Boltzmann constant [J/K]
T = 300.0                  # lattice temperature [K]
V_T = K_B * T / Q          # thermal voltage ~ 0.02585 V
N_C = 2.23e18              # GaN effective conduction-band DOS [cm^-3]
N_V = 4.62e19              # GaN effective valence-band DOS [cm^-3]
J_REF = N_C * V_T          # numerical normalisation of the edge flux
EPS_0_CM = 8.8541878128e-14            # vacuum permittivity [F/cm]
H_REF = 1e-7                           # reference control-volume length [cm]
P_REF = (Q / EPS_0_CM) * N_C * H_REF   # Poisson residual scale [V/cm]

# Band-reference offsets [V]: the simulator's potential carries per-material
# band offsets, so ABSOLUTE densities need n = N_C exp((psi - Phi_n - DELTA_N)/V_T).
# The continuity residual only sees quasi-Fermi gradients and is unaffected;
# the Poisson space-charge term and the device-metric closure need them.
# Calibrated once on ground-truth training fields.
DELTA_N = {"GaN": 2.0772, "AlGaN": 2.1589}
DELTA_P = {"GaN": 1.09, "AlGaN": 1.43}
LOG_DENS_CAP = 57.6        # cap densities at ~1e25 cm^-3 in the log domain

# Node feature columns (see trifield.data.graph_builder)
COL_GAN, COL_ALGAN, COL_GOLD, COL_HFO2, COL_TIN = 3, 4, 5, 6, 7
COL_DOPING, COL_CONTACT, COL_EPS = 8, (10, 11, 12), 15


def contact_mask(x: torch.Tensor) -> torch.Tensor:
    """True for source/drain/gate contact nodes (Dirichlet-pinned)."""
    return (x[:, 10] + x[:, 11] + x[:, 12]) > 0


def _safe_log1p_sq(x: torch.Tensor) -> torch.Tensor:
    """log(1 + x^2) without float32 overflow of x^2."""
    abs_x = x.abs()
    safe = abs_x < 1e18
    sq = torch.where(safe, x * x, torch.ones_like(x))
    return torch.where(safe, torch.log1p(sq), 2.0 * torch.log(abs_x.clamp(min=1e-30)))


class DataLoss(nn.Module):
    """Unweighted per-node MSE in z-score space over non-contact nodes."""

    def forward(self, pred_norm, target_norm, contact=None, valid=None):
        sq = (pred_norm - target_norm) ** 2
        keep = ~contact if contact is not None and (~contact).any() else torch.ones_like(sq[:, 0], dtype=torch.bool)
        if valid is not None:
            m = valid[keep].float()
            return (sq[keep] * m).sum() / m.sum().clamp(min=1.0)
        return sq[keep].mean()


class BalancedDataLoss(nn.Module):
    """Homoscedastic (Kendall et al., 2018) balancing of the data loss over groups:
    one learned log-sigma per (material, field), L = sum_g [exp(-2 s_g) MSE_g / 2 + s_g].
    Groups with a large irreducible error are down-weighted, small-error groups (e.g.
    the oxide) are not drowned by the 85% of nodes that are GaN / AlGaN, and the three
    fields are balanced against each other."""

    N_GROUPS = 6                                   # 5 materials + 'other'

    def __init__(self, n_fields: int = 3):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(self.N_GROUPS, n_fields))

    def forward(self, pred_norm, target_norm, contact, x, weights=None, valid=None):
        mat = torch.where(x[:, 3:8].sum(1) > 0, x[:, 3:8].argmax(1),
                          torch.full_like(x[:, 0], 5, dtype=torch.long))
        keep = ~contact
        sq = (pred_norm - target_norm) ** 2
        w = (weights.unsqueeze(1) if weights is not None else torch.ones_like(sq[:, :1])).expand_as(sq)
        if valid is not None:
            w = w * valid.float()
        sq = sq * w
        total = pred_norm.new_tensor(0.0)
        for gidx in range(self.N_GROUPS):
            k = keep & (mat == gidx)
            if k.any():
                s = self.log_sigma[gidx].clamp(-4, 4)
                denom = w[k].sum(0)
                has = denom > 0
                term = 0.5 * torch.exp(-2 * s) * sq[k].sum(0) / denom.clamp(min=1e-12) + s
                total = total + term[has].sum()
        return total


def channel_weights(y_phys, x, strength: float, offset: float = 2.2, width: float = 0.03):
    """Node weights emphasising the conducting channel and the depletion edge: where the
    conduction band (psi - Phi_n - Delta) is near the Fermi level, the carrier density, and
    with it the Newton residual of the solver, is most sensitive to field error."""
    semi = (x[:, 3] + x[:, 4]) > 0
    act = torch.sigmoid((y_phys[:, 0] - y_phys[:, 1] - offset) / width) * semi
    return 1.0 + strength * act


def gate_weights(x, edge_index, strength: float, hops: int = 2):
    """Node weights 1 + strength on semiconductor nodes within `hops` graph hops of the gate oxide (HfO2):
    the thin gate stack carries ~2x the off-state psi error of other semiconductor nodes and decides whether a
    deep off-state warm start converges. x8 cut that error by 14% (3 seeds) at no cost elsewhere."""
    semi = (x[:, 3] + x[:, 4]) > 0.5
    near = x[:, 6] > 0.5                                     # HfO2 one-hot
    for _ in range(hops):
        nxt = near.clone()
        nxt[edge_index[0][near[edge_index[1]]]] = True
        near = nxt
    return 1.0 + strength * (semi & near).float()


class EdgeDifferenceLoss(nn.Module):
    """MSE of the change of each field along semiconductor edges (z-scored space):
    supervises local structure (e.g. the potential drop at the drain-side gate edge)."""

    def forward(self, pred_norm, target_norm, edge_index, x):
        s, d = edge_index
        semi = ((x[:, 3] + x[:, 4]) > 0)
        k = semi[s] & semi[d]
        dp = pred_norm[d[k]] - pred_norm[s[k]]
        dt = target_norm[d[k]] - target_norm[s[k]]
        return ((dp - dt) ** 2).mean()


class ContinuityResidualLoss(nn.Module):
    """Steady-state current continuity at interior nodes.

    Edge flux (Einstein relation, Boltzmann densities):
        J_n,ij = 0.5 (n_i + n_j) (Phi_n,j - Phi_n,i) / d_ij / J_ref
    The edge list is bidirectional, so a scatter-add over the source index of
    the signed fluxes is the finite-volume divergence at every node.
    """

    def forward(self, pred_phys, edge_index, edge_attr, contact):
        psi, phi_n, phi_p = pred_phys[:, 0], pred_phys[:, 1], pred_phys[:, 2]
        src, dst = edge_index[0], edge_index[1]
        d_ij = edge_attr[:, 0].clamp(min=1e-12)
        n = N_C * torch.exp(((psi - phi_n) / V_T).clamp(-30, 30))
        p = N_V * torch.exp(((phi_p - psi) / V_T).clamp(-30, 30))
        jn = 0.5 * (n[src] + n[dst]) * (phi_n[dst] - phi_n[src]) / d_ij / J_REF
        jp = 0.5 * (p[src] + p[dst]) * (phi_p[dst] - phi_p[src]) / d_ij / J_REF
        n_nodes = pred_phys.size(0)
        div_jn = scatter(jn, src, dim=0, dim_size=n_nodes, reduce="sum")
        div_jp = scatter(jp, src, dim=0, dim_size=n_nodes, reduce="sum")
        interior = ~contact
        if interior.sum() == 0:
            return pred_phys.new_tensor(0.0)
        return (_safe_log1p_sq(div_jn[interior]).mean()
                + _safe_log1p_sq(div_jp[interior]).mean())


class PoissonResidualLoss(nn.Module):
    """Electrostatic Poisson residual, div(eps_r grad psi) + (q/eps_0)(p - n + N_dop) = 0,
    at the same finite-volume level as the continuity term. The space-charge
    term is applied on semiconductor nodes only (dielectrics reduce to
    Laplace); conductor nodes, contact nodes and nodes touching a
    cross-material edge (heterointerface / surface charge) are masked out.
    """

    def forward(self, pred_phys, x, edge_index, edge_attr, contact):
        psi, phi_n, phi_p = pred_phys[:, 0], pred_phys[:, 1], pred_phys[:, 2]
        src, dst = edge_index[0], edge_index[1]
        d_cm = edge_attr[:, 0].clamp(min=1e-12) * 1e-4          # um -> cm
        gan, algan = x[:, COL_GAN], x[:, COL_ALGAN]
        semicon = (gan + algan) > 0
        delta_n = gan * DELTA_N["GaN"] + algan * DELTA_N["AlGaN"]
        delta_p = gan * DELTA_P["GaN"] + algan * DELTA_P["AlGaN"]
        ln_n = (psi - phi_n - delta_n) / V_T + torch.log(torch.tensor(N_C, device=pred_phys.device))
        ln_p = (phi_p - psi - delta_p) / V_T + torch.log(torch.tensor(N_V, device=pred_phys.device))
        n = torch.exp(ln_n.clamp(max=LOG_DENS_CAP))
        p = torch.exp(ln_p.clamp(max=LOG_DENS_CAP))
        f_dop = x[:, COL_DOPING]                                 # sign(N) log10(|N| + 1)
        n_dop = torch.sign(f_dop) * (torch.pow(10.0, f_dop.abs()) - 1.0)
        eps_r = x[:, COL_EPS]
        n_nodes = pred_phys.size(0)
        flux = 0.5 * (eps_r[src] + eps_r[dst]) * (psi[dst] - psi[src]) / d_cm
        div_flux = scatter(flux, src, dim=0, dim_size=n_nodes, reduce="sum")
        h = scatter(d_cm, src, dim=0, dim_size=n_nodes, reduce="mean")
        charge = (Q / EPS_0_CM) * (p - n + n_dop) * semicon.to(pred_phys.dtype)
        residual = (div_flux + charge * h) / P_REF
        conductor = (x[:, COL_GOLD] + x[:, COL_TIN]) > 0
        cross = scatter(edge_attr[:, 4], src, dim=0, dim_size=n_nodes, reduce="sum") > 0
        keep = (~contact) & (~conductor) & (~cross)
        if keep.sum() == 0:
            return pred_phys.new_tensor(0.0)
        return _safe_log1p_sq(residual[keep]).mean()


class CombinedLoss(nn.Module):
    """L = w_data L_data + physics_weight (w_cont L_cont + w_pois L_pois)."""

    def __init__(self, w_data: float = 1.0, w_cont: float = 0.1, w_pois: float = 0.0,
                 balance: bool = False, w_edge: float = 0.0):
        super().__init__()
        self.w_data, self.w_cont, self.w_pois = float(w_data), float(w_cont), float(w_pois)
        self.balance = balance
        self.data_loss = DataLoss()
        self.balanced = BalancedDataLoss() if balance else None
        self.w_edge = float(w_edge)
        self.edge_loss = EdgeDifferenceLoss()
        self.continuity_loss = ContinuityResidualLoss()
        self.poisson_loss = PoissonResidualLoss()

    def forward(self, pred_norm, target_norm, pred_phys, x, edge_index, edge_attr,
                contact, physics_weight: float = 1.0, weights=None, valid=None) -> dict:
        l_mse = self.data_loss(pred_norm, target_norm, contact, valid)
        l_data = self.balanced(pred_norm, target_norm, contact, x, weights, valid) if self.balance else l_mse
        zero = pred_norm.new_tensor(0.0)
        l_cont, l_pois = zero, zero
        if physics_weight > 0:
            l_cont = self.continuity_loss(pred_phys, edge_index, edge_attr, contact)
            if self.w_pois > 0:
                l_pois = self.poisson_loss(pred_phys, x, edge_index, edge_attr, contact)
        total = (self.w_data * l_data
                 + physics_weight * (self.w_cont * l_cont + self.w_pois * l_pois))
        if self.w_edge > 0:
            total = total + self.w_edge * self.edge_loss(pred_norm, target_norm, edge_index, x)
        return {"total": total, "data": l_mse, "continuity": l_cont, "poisson": l_pois}
