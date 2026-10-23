"""Closed-form device-metric *proxies* from the three fields (in-loop estimator).

Given the nine bias snapshots of a design, each with (psi, Phi_n, Phi_p) at
every node, reconstruct carrier densities with a band-offset-corrected
Boltzmann closure and read off the quantities that the acquisition score of
the active-learning loop uses (Sec. 3.4 of the paper):

* ``vth`` [V]: gate bias at which the mean electron density over the GaN
  channel crosses 1e16 cm^-3 (interpolated over the five Vd = 0 snapshots);
* ``id_max``, ``Ron``, ``gm``: electron + hole flux through the drain-boundary
  edges at the high-bias snapshots, width-normalised. These are *relative*
  quantities (no mobility, unit face areas): meaningful for z-scoring and
  ranking candidates within a pool, not as currents in physical units.

The same code runs on predicted and on ground-truth fields. The closure is
exponential in (psi - Phi_n), so it amplifies field error; the device-metric
accuracies reported in the paper were obtained by uploading the predicted
fields into Sentaurus Device, which reads the FoMs from them exactly as from
its own solution (requires a TCAD licence and is not part of this package).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import numpy as np
import torch

from trifield.losses import DELTA_N, DELTA_P, LOG_DENS_CAP, N_C, N_V, Q, V_T
from trifield.params import area_factor

LN_N_C = float(np.log(N_C))
LN_N_V = float(np.log(N_V))


@dataclass
class DeviceMetrics:
    id_max: Optional[float] = None   # drain-flux proxy at (Vg = Vg11, Vd = 12 V), relative units
    vth: Optional[float] = None      # V, gate bias where the mean channel density crosses 1e16 cm^-3
    Ron: Optional[float] = None      # 3 V / flux proxy at (Vg = Vg11, Vd = 3 V), relative units
    gm: Optional[float] = None       # max d(flux proxy)/dVd along the drain sweep, relative units

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def carrier_densities(flds: torch.Tensor, data) -> Tuple[torch.Tensor, torch.Tensor]:
    """Absolute n, p [cm^-3] from (psi, Phi_n, Phi_p) with per-material band references."""
    psi, phi_n, phi_p = flds[:, 0], flds[:, 1], flds[:, 2]
    is_algan = data.x[:, 4] > 0.5
    dn = torch.where(is_algan, torch.full_like(psi, DELTA_N["AlGaN"]),
                     torch.full_like(psi, DELTA_N["GaN"]))
    dp = torch.where(is_algan, torch.full_like(psi, DELTA_P["AlGaN"]),
                     torch.full_like(psi, DELTA_P["GaN"]))
    n = torch.exp(((psi - phi_n - dn) / V_T + LN_N_C).clamp(max=LOG_DENS_CAP))
    p = torch.exp(((phi_p - psi - dp) / V_T + LN_N_V).clamp(max=LOG_DENS_CAP))
    return n, p


def _channel_mask(data) -> torch.Tensor:
    is_gan = data.x[:, 3] > 0.5
    is_contact = (data.x[:, 10] + data.x[:, 11] + data.x[:, 12]) > 0
    return is_gan & ~is_contact


def drain_current(flds: torch.Tensor, data) -> float:
    """Electron + hole flux through the drain-boundary edges (relative units)."""
    phi_n, phi_p = flds[:, 1], flds[:, 2]
    n, p = carrier_densities(flds, data)
    is_drain = data.x[:, 11] > 0.5
    src, dst = data.edge_index[0], data.edge_index[1]
    d_ij = data.edge_attr[:, 0].clamp(min=1e-12)
    boundary = (is_drain[src] != is_drain[dst]) & (~is_drain[src])   # channel -> drain
    s, d, dd = src[boundary], dst[boundary], d_ij[boundary]
    if s.numel() == 0:
        return 0.0
    jn = 0.5 * (n[s] + n[d]) * (phi_n[d] - phi_n[s]) / dd
    jp = 0.5 * (p[s] + p[d]) * (phi_p[d] - phi_p[s]) / dd
    total = Q * float(jn.sum() + jp.sum())
    af = area_factor(data.params) if getattr(data, "params", None) else 1.0
    return abs(total) * af


def mean_channel_density(flds: torch.Tensor, data) -> float:
    n, _ = carrier_densities(flds, data)
    ch = _channel_mask(data)
    return float(n[ch].mean()) if ch.any() else 0.0


def extract_device_metrics(snapshots: List[Tuple]) -> DeviceMetrics:
    """FoMs of one design from ``[(Data, fields (N, 3)), ...]`` over its bias snapshots."""
    out = DeviceMetrics()
    if not snapshots:
        return out
    data0 = snapshots[0][0]
    vg11 = float(data0.params["Vg11"]) if getattr(data0, "params", None) else 5.0
    by_bias = {(round(float(d.vg), 2), round(float(d.vd), 2)): (d, f) for d, f in snapshots}

    key = (round(vg11, 2), 12.0)
    if key in by_bias:
        out.id_max = drain_current(by_bias[key][1], by_bias[key][0])

    key = (round(vg11, 2), 3.0)
    if key in by_bias:
        i_lin = drain_current(by_bias[key][1], by_bias[key][0])
        if i_lin > 1e-12:
            out.Ron = 3.0 / i_lin

    curve = [(vd, drain_current(*reversed(by_bias[(round(vg11, 2), vd)])))
             for vd in (3.0, 6.0, 9.0, 12.0) if (round(vg11, 2), vd) in by_bias]
    if len(curve) >= 2:
        slopes = [abs((curve[i][1] - curve[i - 1][1]) / (curve[i][0] - curve[i - 1][0]))
                  for i in range(1, len(curve))]
        out.gm = max(slopes) * 1e3

    dens = [(round(f * vg11, 2), mean_channel_density(*reversed(by_bias[(round(f * vg11, 2), 0.0)])))
            for f in (0.2, 0.4, 0.6, 0.8, 1.0) if (round(f * vg11, 2), 0.0) in by_bias]
    if len(dens) >= 2:
        thr = 1e16
        for i in range(1, len(dens)):
            (v_lo, n_lo), (v_hi, n_hi) = dens[i - 1], dens[i]
            if n_lo <= thr <= n_hi and n_hi > n_lo:
                out.vth = v_lo + (thr - n_lo) / (n_hi - n_lo) * (v_hi - v_lo)
                break
        else:
            out.vth = 0.0 if dens[0][1] > thr else (vg11 if dens[-1][1] < thr else None)
    return out
