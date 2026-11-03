"""Exact-operator physics loss: Sentaurus' discrete drift-diffusion residuals of a predicted field triple.

The training-time residual of ``trifield.losses.ContinuityResidualLoss`` is not Sentaurus' discretisation (Boltzmann
densities without the band reference, 1/d edge weights, arithmetic-mean fluxes, constant mobility, oxide / metal
placeholders): it does not vanish at Sentaurus' own solution and its minimum lies volts away from it. This module builds
the residual on ``trifield.physics.equations`` instead (the operator that ``Solver.refine`` solves; ~1e-6 relative on
the float32 labels), per design from the per-mesh exports (``export_mesh_data``), differentiable in torch (float64):

* boundary: hard Dirichlet values from the deck's boundary rules (gate psi = Vg + offset, Gold psi = V + offset, ohmic
  semiconductor vertices: charge neutrality at Phi = V, AlGaN(x=0) corner rule, Phi = V at every metal vertex), so the
  physics term sees the applied bias even on unlabelled meshes; the network's own values there get no physics gradient;
* Poisson rows (``in_eq & ~dirichlet``): r / diag, V-equivalent; diag = the flux-operator diagonal (``pdiag``, the
  refinement's merit) or, with ``poisson_norm="jacobi"``, pdiag + q M |d SC / d psi| (the local Newton correction in volts:
  bounded in the 2DEG, where the space-charge derivative dominates);
* electron continuity rows (``semi & ~dirichlet``): r / gross with the gross one-directional SG flows (detached; the
  refinement's row scaling): |r / gross| <= 1, about the local Phi_n correction in units of k_B T / q; carrier-free rows
  (n < freeze_n, trap vertices below 1e-3 n1; decided on the detached prediction) are dropped, as the refinement freezes
  them; mobility detached by default (Caughey-Thomas with beta < 1 has a cusp at zero driving force);
* optional hole continuity rows (same scaling, p < freeze_p dropped).

Loss = w_P mean rho(r_P) + w_n mean rho(r_n) + w_p mean rho(r_p), rho(x) = x^2 (``rho="sq"``) or s^2 log(1 + x^2 / s^2)
(``rho="log"``, robust to a few huge rows early on), averaged over the graphs that have an export.

Use ``norm_from="ref"``: normalisers and carrier-free mask from fixed reference fields (labels, or the first prediction of
an unlabelled item). With the default "pred" they follow the prediction and are detached, which drops a gradient term as
large as the kept one on badly balanced rows; descent then raises the loss (paired fine-tunes, palto scripts/al/RESULTS.md
section 9: weight 0.1 doubled the field error). Even with a correct gradient the loss did not make fields more accurate;
the equations add information through ``Solver.refine`` instead (``trifield.pseudolabel``).
"""
from __future__ import annotations

import collections
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from trifield.physics.equations import DT, Equations, bern
from trifield.physics.fermi import KT, r_of_eta_t
from trifield.physics.mesh_data import EDGES


@dataclass
class PhysicsLossConfig:
    w_poisson: float = 1.0
    w_electron: float = 1.0
    w_hole: float = 0.0
    poisson_norm: str = "jacobi"          # "pdiag" (refinement merit) or "jacobi"
    rho: str = "log"                      # "sq" or "log" (both terms unless overridden below)
    rho_poisson: str | None = None        # e.g. "sq": no saturation of the Poisson rows (a saturated Poisson penalty
                                          # lets an unanchored prediction deplete the device and freeze every
                                          # continuity row: physics-only test-time training found that state)
    freeze_from_labels: bool = True       # labelled graphs: carrier-free rows from the label fields, not the prediction
    # where the row normalisers (Jacobi diagonal, gross flows) and the carrier-free mask come from: "pred" = the current
    # prediction, detached (the gradient then ignores d(normaliser)/d(fields), which is as large as the kept term on rows
    # with imbalance ~1: the step is not a descent direction of the reported loss); "ref" = fixed reference fields (the
    # labels, or for unlabelled items the first prediction seen), i.e. a constant-weight least-squares residual
    norm_from: str = "pred"
    scale_poisson: float = 0.1            # rho scale s [V-equivalent] of the Poisson rows
    scale_cont: float = 0.1               # rho scale of the continuity rows (|r / gross| <= 1)
    freeze_n: float = 1e-2                # cm^-3, as RefineConfig.freeze_n
    freeze_p: float = 1e-2
    lag_mobility: bool = True
    recombination: bool = True


class MeshPhysics:
    """Torch tensors of one design's MeshData on one device and the residuals of a field triple."""

    def __init__(self, md, device="cpu"):
        self.md = md
        self.dev = torch.device(device)
        eq = Equations(md, self.dev)
        if getattr(md, "trap_factors", None) is not None:
            eq.set_traps(md.trap_factors)
        self.eq = eq
        t = lambda a, dt=DT: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
        Kc = md.K.tocoo()
        self.K = torch.sparse_coo_tensor(np.vstack([Kc.row, Kc.col]), Kc.data, (md.n, md.n), dtype=DT,
                                         device=self.dev).coalesce()
        self.U = torch.as_tensor(np.where(md.in_eq & ~md.dirichlet)[0], device=self.dev)
        self.V = torch.as_tensor(np.where(md.semi & ~md.dirichlet & (md.Msemi > 0))[0], device=self.dev)
        self.D = t(md.dirichlet, torch.bool)
        self.semi = t(md.semi, torch.bool)
        self.pdiag = eq.pdiag
        self.qM = eq.qM
        # hole element data (semiconductor elements)
        self.MLp = eq.MLp
        self._bc = {}

    # ------------------------------------------------------------------ boundary rules
    def _offsets(self):
        """Dirichlet psi = (electrode bias of the vertex) + offset. Every rule is bias-shift invariant: gate metal
        Vg + gate_offset, Gold V + gold_offset, AlGaN(x=0) ohmic corners V + x0_offset, other ohmic semiconductor
        vertices charge neutrality at Phi = V, whose space charge depends on psi - V only. One Newton per design."""
        if getattr(self, "_off", None) is not None:
            return self._off
        md, eq = self.md, self.eq
        n = md.n
        off = np.full(n, np.nan)
        tin = md.metal_material == "TiN"
        off[tin] = md.gate_offset
        gold_only = md.metal_only & (md.metal_material == "Gold")
        off[gold_only] = md.gold_offset
        x0 = md.dirichlet & md.x0 & ~tin
        off[x0] = md.x0_offset
        s = md.dirichlet & md.semi & ~tin & ~md.x0
        if s.any():                                              # pointwise charge neutrality at Phi = 0
            z = torch.zeros(n, dtype=DT, device=self.dev)
            x = torch.as_tensor(md.C0 - np.asarray(md.chi), dtype=DT, device=self.dev)
            sm = torch.as_tensor(s, device=self.dev)
            with torch.enable_grad():
                for _ in range(200):
                    P = x.detach().clone().requires_grad_(True)
                    sc = eq.space_charge(P, z, z)
                    d, = torch.autograd.grad(sc.sum(), P)
                    st = torch.where(sm, torch.clamp(-sc.detach() / torch.where(d != 0, d, -torch.ones_like(d)), -0.5, 0.5),
                                     torch.zeros_like(x))
                    x = (x + st).detach()
                    if float(st.abs().max()) < 1e-13:
                        break
            off[s] = x.cpu().numpy()[s]
        fixed = np.isfinite(off)
        self._off = (torch.as_tensor(fixed, device=self.dev), torch.as_tensor(np.nan_to_num(off), dtype=DT, device=self.dev))
        return self._off

    def boundary(self, vg: float, vd: float):
        """(fixed mask, Dirichlet psi, electrode bias per vertex) at (vg, vd)."""
        key = (round(float(vg), 9), round(float(vd), 9))
        if key in self._bc:
            return self._bc[key]
        fixed, off = self._offsets()
        V = torch.as_tensor(self.md.electrode_bias(vg, vd), dtype=DT, device=self.dev)
        out = (fixed, V + off, V)
        if len(self._bc) > 64:
            self._bc.clear()
        self._bc[key] = out
        return out

    def apply_boundary(self, fields, vg, vd):
        """(n,3) -> psi, Phi_n, Phi_p with the Dirichlet values imposed (float64)."""
        f = fields.to(DT)
        fixed, pd, V = self.boundary(vg, vd)
        psi = torch.where(fixed, pd, f[:, 0])
        phin = torch.where(self.D, V, f[:, 1])
        phip = torch.where(self.D, V, f[:, 2])
        z = torch.zeros_like(psi)
        return psi, torch.where(self.semi, phin, z), torch.where(self.semi, phip, z)

    # ------------------------------------------------------------------ holes (differentiable, element-local)
    def hole_residual(self, psi, phin, phip, recombination=True, lag=True):
        eq = self.eq
        Ec, Ev, n, p, gn, gp = eq.bands(psi, phin, phip)
        w = psi + eq.chi + eq.Eg - KT * torch.log(gp)
        mu = eq.mobility_e(psi[eq.Ts], phip[eq.Ts], holes=True)
        if lag:
            mu = mu.detach()
        r = torch.zeros(eq.n, dtype=DT, device=self.dev); g = torch.zeros_like(r)
        for k, (a, b) in enumerate(EDGES):
            ia, ib = eq.Ts[:, a], eq.Ts[:, b]
            d = (w[ib] - w[ia]) / KT
            c = mu * KT * eq.Cs[:, k] * 1e-4
            fa, fb = p[ia] * bern(d), p[ib] * bern(-d)               # hole flow a -> b = c (fa - fb)
            f = c * (fa - fb)
            r = r.index_add(0, ia, -f).index_add(0, ib, f)
            gg = (c.abs() * (fa + fb)).detach()
            g = g.index_add(0, ia, gg).index_add(0, ib, gg)
        if recombination:
            RV = eq.Msemi * eq.recombination(psi, phin, phip)
            r = r - RV; g = g + RV.detach().abs()
        return r, g, p

    # ------------------------------------------------------------------ residuals
    def residuals(self, fields, vg, vd, cfg: PhysicsLossConfig | None = None, holes: bool | None = None, ref=None):
        """Normalised residual rows of a (n,3) field triple: dict(poisson, electron[, hole], n_frozen). ``ref``: fields
        that decide the carrier-free (dropped) continuity rows instead of the prediction (e.g. the labels), and with
        cfg.norm_from == "ref" also the row normalisers."""
        cfg = cfg or PhysicsLossConfig()
        eq = self.eq
        psi, phin, phip = self.apply_boundary(fields, vg, vd)
        use_ref = ref is not None and cfg.norm_from == "ref"
        npsi, nphin, nphip = (self.apply_boundary(ref.detach(), vg, vd) if use_ref
                              else (psi.detach(), phin.detach(), phip.detach()))
        # Poisson
        sc = eq.space_charge(psi, phin, phip)
        rp = torch.sparse.mm(self.K, psi[:, None])[:, 0] + self.qM * sc
        if cfg.poisson_norm == "jacobi":
            with torch.enable_grad():
                P = npsi.detach().clone().requires_grad_(True)
                dsc, = torch.autograd.grad(eq.space_charge(P, nphin, nphip).sum(), P)
            diag = self.pdiag + self.qM * dsc.abs()
        else:
            diag = self.pdiag
        out = {"poisson": rp[self.U] / diag[self.U]}
        # electron continuity
        er, em = eq.electron_elem(psi[eq.Ts], phin[eq.Ts], lag=cfg.lag_mobility)
        z = torch.zeros(eq.n, dtype=DT, device=self.dev)
        r = z.index_add(0, eq.Ts.reshape(-1), er.reshape(-1))
        if use_ref:
            with torch.no_grad():
                _, em = eq.electron_elem(npsi[eq.Ts], nphin[eq.Ts], lag=True)
        m = z.index_add(0, eq.Ts.reshape(-1), em.reshape(-1).detach())
        if cfg.recombination:
            RV = eq.Msemi * eq.recombination(psi, phin, phip)
            r = r - RV
            with torch.no_grad():
                m = m + (eq.Msemi * eq.recombination(npsi, nphin, nphip)).abs() if use_ref else m + RV.detach().abs()
        V = self.V
        with torch.no_grad():
            if ref is not None:
                Ec, Ev, n, p, gn, gp = eq.bands(*self.apply_boundary(ref.detach(), vg, vd))
            else:
                Ec, Ev, n, p, gn, gp = eq.bands(npsi, nphin, nphip)
            thr = torch.full_like(n, cfg.freeze_n)
            for idx, conc, emid, cc, K in eq.traps:
                Et = (Ec[idx] + Ev[idx]) / 2 + KT / 2 * torch.log(eq.Nv[idx] / eq.Nc[idx]) + emid
                n1 = gn[idx] * eq.Nc[idx] * torch.exp((Et - Ec[idx]) / KT)
                thr[idx] = torch.minimum(thr[idx], 1e-3 * n1)
            act = n[V] >= thr[V]
            cs = torch.clamp(m[V], min=1e-30 + 1e-12 * float(m[V].median()))
        out["electron"] = (r[V] / cs)[act]
        out["n_frozen"] = int((~act).sum())
        if holes if holes is not None else cfg.w_hole > 0:
            rh, gh, p = self.hole_residual(psi, phin, phip, cfg.recombination, cfg.lag_mobility)
            with torch.no_grad():
                acth = p[V] >= cfg.freeze_p
                csh = torch.clamp(gh[V], min=1e-30 + 1e-12 * float(gh[V].median()))
            out["hole"] = (rh[V] / csh)[acth]
        return out


def rho(x, kind: str, s: float):
    if kind == "sq":
        return x * x
    return s * s * torch.log1p((x / s) ** 2)


class PhysicsRegistry:
    """design name -> MeshPhysics, from per-mesh exports under one or more roots
    (``<root>/<design>/twin/physics/mesh_data.pkl`` + optional ``trap_factors.pkl``). MeshData are loaded lazily and
    kept on the CPU; device copies live in an LRU of ``max_device`` designs."""

    def __init__(self, roots, device="cpu", max_device: int = 128):
        self.dev = torch.device(device)
        self.paths: dict[str, Path] = {}
        for root in roots:
            for p in sorted(Path(root).glob("*/twin/physics/mesh_data.pkl")):
                self.paths.setdefault(p.parent.parent.parent.name, p.parent)
        self._md: dict[str, object] = {}
        self._dev = collections.OrderedDict()
        self.max_device = max_device

    def __contains__(self, design: str) -> bool:
        return design in self.paths

    def __len__(self):
        return len(self.paths)

    def mesh_data(self, design: str):
        if design not in self._md:
            d = self.paths[design]
            md = pickle.loads((d / "mesh_data.pkl").read_bytes())
            tf = d / "trap_factors.pkl"
            md.trap_factors = pickle.loads(tf.read_bytes()) if tf.exists() else None
            self._md[design] = md
        return self._md[design]

    def get(self, design: str) -> MeshPhysics:
        if design in self._dev:
            self._dev.move_to_end(design)
            return self._dev[design]
        mp = MeshPhysics(self.mesh_data(design), self.dev)
        self._dev[design] = mp
        while len(self._dev) > self.max_device:
            self._dev.popitem(last=False)
        return mp


class ExactPhysicsLoss(torch.nn.Module):
    """Mean over the graphs of a batch that have an export of w_P rho(Poisson) + w_n rho(electron) [+ w_p rho(hole)]."""

    def __init__(self, registry: PhysicsRegistry, cfg: PhysicsLossConfig | None = None):
        super().__init__()
        self.reg = registry
        self.cfg = cfg or PhysicsLossConfig()
        self._ref: dict = {}                 # norm_from="ref", unlabelled items: first prediction seen (CPU, float32)

    def graph_terms(self, pred, design, vg, vd, ref=None):
        cfg = self.cfg
        mp = self.reg.get(design)
        res = mp.residuals(pred, vg, vd, cfg, ref=ref)
        terms = {"poisson": rho(res["poisson"], cfg.rho_poisson or cfg.rho, cfg.scale_poisson).mean()}
        e = res["electron"]
        terms["electron"] = rho(e, cfg.rho, cfg.scale_cont).mean() if len(e) else pred.new_zeros((), dtype=DT)
        if cfg.w_hole > 0:
            h = res["hole"]
            terms["hole"] = rho(h, cfg.rho, cfg.scale_cont).mean() if len(h) else pred.new_zeros((), dtype=DT)
        tot = cfg.w_poisson * terms["poisson"] + cfg.w_electron * terms["electron"] + cfg.w_hole * terms.get("hole", 0.0)
        with torch.no_grad():
            merit = torch.sqrt(torch.cat([res["poisson"].detach() / 1.0, e.detach()]).pow(2).mean())
        return tot, terms, float(merit)

    def forward(self, pred_phys, batch):
        """pred_phys: (N, 3) for the whole batch; batch: PyG Batch with ptr, sim_name, vg, vd."""
        names = batch.sim_name if isinstance(batch.sim_name, (list, tuple)) else [batch.sim_name]
        vgs = batch.vg.tolist() if torch.is_tensor(batch.vg) else list(np.atleast_1d(batch.vg))
        vds = batch.vd.tolist() if torch.is_tensor(batch.vd) else list(np.atleast_1d(batch.vd))
        ptr = batch.ptr.tolist() if hasattr(batch, "ptr") and batch.ptr is not None else [0, pred_phys.shape[0]]
        total, n, acc = None, 0, collections.defaultdict(float)
        merits = []
        for i, d in enumerate(names):
            if d not in self.reg:
                continue
            ref = None
            labelled = False
            if hasattr(batch, "y") and batch.y is not None:
                v = getattr(batch, "y_valid", None)
                labelled = v is None or bool(v[ptr[i]:ptr[i + 1]].all())
            if labelled and (self.cfg.freeze_from_labels or self.cfg.norm_from == "ref"):
                ref = batch.y[ptr[i]:ptr[i + 1]]
            elif not labelled and self.cfg.norm_from == "ref":
                key = (d, round(float(vgs[i]), 6), round(float(vds[i]), 6))
                if key not in self._ref:
                    self._ref[key] = pred_phys[ptr[i]:ptr[i + 1]].detach().float().cpu()
                ref = self._ref[key].to(pred_phys.device)
            tot, terms, merit = self.graph_terms(pred_phys[ptr[i]:ptr[i + 1]], d, vgs[i], vds[i], ref)
            total = tot if total is None else total + tot
            n += 1
            for k, v in terms.items():
                acc[k] += float(v.detach())
            merits.append(merit)
        if total is None:
            return pred_phys.new_zeros(()), {"n_graphs": 0}
        stats = {k: v / n for k, v in acc.items()}
        stats["n_graphs"] = n
        stats["merit"] = float(np.mean(merits))
        return (total / n).to(pred_phys.dtype), stats
