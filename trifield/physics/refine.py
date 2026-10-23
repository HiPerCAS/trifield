"""Refinement of an initial guess on Sentaurus' exact discrete equations, and terminal currents.

refine(guess, vg, vd, solver):
  1. boundary rules (gate metal, Ohmic Gold/semiconductor vertices, AlGaN(x=0) corner rule, Gold bulk)
  2. damped Gummel: Poisson Newton (Phi frozen) <-> linear SG solve for n (psi, mobility, gamma lagged),
     Phi_n steps capped at `damp`, under-relaxed by `relax`, clipped to the electrode range
  3. joint (psi, Phi_n) Newton on Poisson + electron continuity (row-scaled, backtracking line search)
  4. Phi_p from the (linear, lagged) hole continuity incl. SRH and interface-trap recombination
Holes carry no charge of consequence here but Sentaurus' convergence depends strongly on Phi_p.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from trifield.physics.equations import DT, Equations
from trifield.physics.fermi import KT
from trifield.physics.linalg import Backend
from trifield.physics.mesh_data import METAL_RESISTANCE, MeshData, Q


@dataclass
class RefineConfig:
    gummel_iters: int = 25
    gummel_damp: float = 0.05
    gummel_relax: float = 0.5
    poisson_maxit: int = 50
    poisson_tol: float = 1e-10       # inner Poisson-Newton step tolerance [V]
    newton_maxit: int = 120
    newton_clip: float = 0.5
    newton_tol: float = 1e-9
    hole_iters: int = 8
    phin_margin: float = 0.05
    ptc: float = 0.0                 # pseudo-transient continuation: J + delta*I (scaled), delta *= merit ratio
    cont_floor: float = 1e-12        # continuity row scale floor (x median flux) for the merit
    restarts: int = 8                # joint-Newton stalls -> extra Gummel sweeps, then Newton again
    restart_gummel: int = 10
    stall_window: int = 6            # Newton iterations without a 30% merit reduction = stall


@dataclass
class RefineResult:
    psi: np.ndarray
    phin: np.ndarray
    phip: np.ndarray
    converged: bool
    gummel_iters: int
    newton_iters: int
    merit: float
    poisson_rms: float
    continuity_rms: float
    time: dict = field(default_factory=dict)

    def fields(self):
        return {"ElectrostaticPotential": self.psi, "eQuasiFermiPotential": self.phin, "hQuasiFermiPotential": self.phip}


class Solver:
    """Per-design refinement engine (holds device tensors and sparse operators)."""

    def __init__(self, md: MeshData, device="cpu", config: RefineConfig | None = None):
        self.md = md
        self.cfg = config or RefineConfig()
        self.bk = Backend(device)
        self.dev = self.bk.device
        self.eq = Equations(md, self.dev)
        self.n = md.n
        sp = self.bk.sp
        Kc = md.K.tocoo()
        self.K = self.bk.coo(torch.as_tensor(Kc.row, device=self.dev), torch.as_tensor(Kc.col, device=self.dev),
                             torch.as_tensor(Kc.data, device=self.dev), (self.n, self.n))
        self.Kt = torch.sparse_coo_tensor(np.vstack([Kc.row, Kc.col]), Kc.data, (self.n, self.n), dtype=DT,
                                          device=self.dev).coalesce()
        self.U = np.where(md.in_eq & ~md.dirichlet)[0]                               # Poisson unknowns
        self.V = np.where(md.semi & ~md.dirichlet & (md.Msemi > 0))[0]               # continuity unknowns
        self.Dn = np.where(md.semi & md.dirichlet)[0]
        self.PU = self._sel(self.U); self.PV = self._sel(self.V); self.PD = self._sel(self.Dn)
        self.Ut = torch.as_tensor(self.U, device=self.dev); self.Vt = torch.as_tensor(self.V, device=self.dev)
        self.Dt = torch.as_tensor(self.Dn, device=self.dev)
        pos = -np.ones(self.n, int); pos[self.V] = np.arange(len(self.V))
        self.UinV = np.where(pos[self.U] >= 0)[0]; self.UinV_col = pos[self.U[self.UinV]]
        self.KUU = self.PU @ self.K @ self.PU.T                                       # constant Poisson block

    def _sel(self, idx):
        r = torch.arange(len(idx), device=self.dev); c = torch.as_tensor(idx, device=self.dev)
        return self.bk.coo(r, c, torch.ones(len(idx), dtype=DT, device=self.dev), (len(idx), self.n))

    def _split(self, V, dens, Dn, tiny=1e-250, huge=float("inf")):
        """Unknowns of a linear density solve: vertices with a representable density; the rest (n or p
        underflowing, e.g. Schottky-like Gold corners) are held fixed like Dirichlet nodes."""
        ok = (dens[V] > tiny) & (dens[V] < huge)
        act = V[ok]
        fix = torch.cat([Dn, V[~ok]])
        return act, fix

    def t(self, a):
        return torch.as_tensor(np.asarray(a, float), dtype=DT, device=self.dev)

    # ------------------------------------------------------------------ pieces
    def charge_derivs(self, psi, phin, phip):
        P = psi.detach().clone().requires_grad_(True); F = phin.detach().clone().requires_grad_(True)
        sc = self.eq.space_charge(P, F, phip)
        gp, gf = torch.autograd.grad(sc.sum(), (P, F))
        return sc.detach(), gp, gf

    def poisson_res(self, psi, phin, phip):
        return torch.sparse.mm(self.Kt, psi[:, None])[:, 0] + self.eq.qM * self.eq.space_charge(psi, phin, phip)

    def boundary(self, guess, vg, vd):
        md = self.md
        V = md.electrode_bias(vg, vd)
        g = np.asarray(guess, float)
        phin = g[:, 1].copy(); phip = g[:, 2].copy()
        phin[md.dirichlet] = V[md.dirichlet]; phip[md.dirichlet] = V[md.dirichlet]
        pd = np.full(self.n, np.nan); pd[md.dirichlet] = g[md.dirichlet, 0]
        tin = md.metal_material == "TiN"
        pd[tin] = vg + md.gate_offset
        s = md.dirichlet & md.semi & ~tin & ~md.x0
        x = self.t(g[:, 0]); Pn, Pp = self.t(phin), self.t(phip); sm = self.t(s) > 0
        for _ in range(200):                                    # pointwise neutrality SC(psi; Phi) = 0
            sc, d, _ = self.charge_derivs(x, Pn, Pp)
            st = torch.where(sm, torch.clamp(-sc / torch.where(d != 0, d, -torch.ones_like(d)), -0.5, 0.5), torch.zeros_like(x))
            x = x + st
            if float(st.abs().max()) < 1e-13:
                break
        pd[s] = x.cpu().numpy()[s]
        x0 = md.dirichlet & md.x0 & ~tin
        pd[x0] = phin[x0] + md.x0_offset
        gold_only = md.metal_only & (md.metal_material == "Gold")
        pd[gold_only] = V[gold_only] + md.gold_offset
        return pd, phin, phip

    def poisson_newton(self, psi, phin, phip, maxit):
        bk, U = self.bk, self.Ut
        KUU = self.KUU
        def merit(p):
            r = self.poisson_res(p, phin, phip)[U] / self.eq.pdiag[U]
            return float(torch.sqrt((r ** 2).mean()))
        m = merit(psi); it = 0
        for it in range(maxit):
            sc, d, _ = self.charge_derivs(psi, phin, phip)
            F = (torch.sparse.mm(self.Kt, psi[:, None])[:, 0] + self.eq.qM * sc)[U]
            J = KUU + bk.sp.diags(bk.arr((self.eq.qM * d)[U]))
            delta = bk.tens(bk.solve(J, bk.arr(-F), key="poisson")).to(DT)
            if not bool(torch.isfinite(delta).all()):
                break
            delta = torch.clamp(delta, -1.0, 1.0)
            lam = 1.0
            while True:
                trial = psi.clone(); trial[U] += lam * delta
                mt = merit(trial)
                if mt < m or lam < 1e-4: break
                lam /= 2
            psi, m = trial, mt
            if float((lam * delta).abs().max()) < self.cfg.poisson_tol:
                break
        return psi, m

    def gummel(self, psi, phin, phip, bounds, iters=None):
        cfg, bk, eq = self.cfg, self.bk, self.eq
        V, Dn = self.Vt, self.Dt
        hist = []
        for it in range(cfg.gummel_iters if iters is None else iters):
            psi, mp = self.poisson_newton(psi, phin, phip, cfg.poisson_maxit)
            r, c, v, nfull = eq.electron_sg_operator(psi, phin)
            A = bk.coo(r, c, v, (self.n, self.n))
            act, fix = self._split(V, nfull, Dn)
            Pa, Pf = self._sel(act), self._sel(fix)
            s = nfull[act]
            rhs = -((Pa @ A @ Pf.T) @ bk.arr(nfull[fix]))
            y = bk.tens(bk.solve((Pa @ A @ Pa.T) @ bk.sp.diags(bk.arr(s)), rhs, key="sg")).to(DT)
            if not bool(torch.isfinite(y).all()):
                break
            nn = nfull.clone(); nn[act] = torch.clamp(y * s, min=1e-300)
            new = eq.phin_from_n(psi, nn)
            step = torch.clamp(cfg.gummel_relax * (new[act] - phin[act]), -cfg.gummel_damp, cfg.gummel_damp)
            if not bool(torch.isfinite(step).all()):
                break
            phin = phin.clone(); phin[act] = torch.clamp(phin[act] + step, *bounds)
            hist.append(float(step.abs().max()))
            if hist[-1] < 1e-7:
                break
        return psi, phin, hist

    def _eval(self, psi, phin, phip, jac):
        eq = self.eq
        if jac:
            sc, dps, dpf = self.charge_derivs(psi, phin, phip)
            Fp = torch.sparse.mm(self.Kt, psi[:, None])[:, 0] + eq.qM * sc
            r, m, rows, cols, vp, vf = eq.electron_jacobian(psi, phin)
            return dict(Fp=Fp, dps=dps, dpf=dpf, Fc=r, mc=m, rows=rows, cols=cols, vp=vp, vf=vf)
        with torch.no_grad():
            Fp = self.poisson_res(psi, phin, phip)
            r, m = eq.electron(psi, phin)
        return dict(Fp=Fp, Fc=r, mc=m)

    def joint_newton(self, psi, phin, phip, bounds, maxit=None, stall_check=False):
        cfg, bk, eq = self.cfg, self.bk, self.eq
        maxit = cfg.newton_maxit if maxit is None else maxit
        U, V = self.Ut, self.Vt
        KUU = self.KUU
        def merit(o, cs):
            a = o["Fp"][U] / eq.pdiag[U]; b = o["Fc"][V] / cs
            return (float(torch.sqrt((torch.cat([a, b]) ** 2).mean())), float(torch.sqrt((a ** 2).mean())),
                    float(torch.sqrt((b ** 2).mean())))
        hist = []; good = (psi.clone(), phin.clone()); delta = cfg.ptc
        for it in range(maxit + 1):
            o = self._eval(psi, phin, phip, True)
            cs = torch.clamp(o["mc"][V], min=1e-30 + cfg.cont_floor * float(o["mc"][V].median()))
            m, mp, mc = merit(o, cs)
            if not np.isfinite(m):
                psi, phin = good; break
            good = (psi.clone(), phin.clone()); hist.append((m, mp, mc))
            if it == maxit or m < cfg.newton_tol:
                break
            w = cfg.stall_window
            if stall_check and len(hist) > w and hist[-1][0] > 0.7 * hist[-1 - w][0]:
                break
            Jc_p = bk.coo(o["rows"], o["cols"], o["vp"], (self.n, self.n))
            Jc_f = bk.coo(o["rows"], o["cols"], o["vf"], (self.n, self.n))
            rp = bk.sp.diags(bk.arr(1 / eq.pdiag[U])); rc = bk.sp.diags(bk.arr(1 / cs))
            Jpp = KUU + bk.sp.diags(bk.arr((eq.qM * o["dps"])[U]))
            Jpf = bk.coo(torch.as_tensor(self.UinV, device=self.dev), torch.as_tensor(self.UinV_col, device=self.dev),
                         (eq.qM * o["dpf"])[U][self.UinV], (len(self.U), len(self.V)))
            Jcp = self.PV @ Jc_p @ self.PU.T; Jcf = self.PV @ Jc_f @ self.PV.T
            J = bk.sp.bmat([[rp @ Jpp, rp @ Jpf], [rc @ Jcp, rc @ Jcf]]).tocsr()
            if delta > 0:
                if len(hist) > 1:
                    delta = max(delta * hist[-1][0] / hist[-2][0], 0.0)
                J = J - delta * bk.sp.identity(J.shape[0], format="csr")
            rhs = torch.cat([-o["Fp"][U] / eq.pdiag[U], -o["Fc"][V] / cs])
            dx = bk.tens(bk.solve(J, bk.arr(rhs), key="joint")).to(DT)
            if not bool(torch.isfinite(dx).all()):
                break
            dpsi, dphi = dx[:len(self.U)], dx[len(self.U):]
            smax = float(dx.abs().max()); lam = min(1.0, cfg.newton_clip / smax) if smax > 0 else 1.0
            while True:
                tp, tf = psi.clone(), phin.clone(); tp[U] += lam * dpsi; tf[V] = torch.clamp(tf[V] + lam * dphi, *bounds)
                ot = self._eval(tp, tf, phip, False)
                if merit(ot, cs)[0] < m or lam < 1e-4: break
                lam /= 2
            psi, phin = tp, tf
            if smax * lam < 1e-12: break
        return psi, phin, hist

    def holes(self, psi, phin, phip):
        bk, eq = self.bk, self.eq
        V, Dn = self.Vt, self.Dt
        for it in range(self.cfg.hole_iters):
            r, c, v, a, b, p = eq.hole_operator(psi, phin, phip)
            A = bk.coo(r, c, v, (self.n, self.n)) - bk.sp.diags(bk.arr(eq.Msemi * a))
            act, fix = self._split(V, p, Dn, tiny=0.0)
            Pa, Pf = self._sel(act), self._sel(fix)
            s = p[act]
            rhs = -((Pa @ A @ Pf.T) @ bk.arr(p[fix])) - bk.arr((eq.Msemi * b)[act])
            M = (Pa @ A @ Pa.T) @ bk.sp.diags(bk.arr(s))
            if bk.gpu:
                # the p-scaled hole system is badly scaled (entries ~1e-17..1e-50): cuDSS with row
                # equilibration returns Phi_p up to ~1e150 V on some meshes, SuperLU does not -> CPU solve
                import scipy.sparse.linalg as spla
                y = torch.as_tensor(spla.spsolve(M.get().tocsc(), rhs.get()), device=self.dev).to(DT)
            else:
                y = bk.tens(bk.solve(M, rhs, key="holes")).to(DT)
            if not bool(torch.isfinite(y).all()):
                break
            pn = p.clone(); pn[act] = torch.clamp(y * s, min=1e-300)
            new = eq.phip_from_p(psi, pn)
            d = float((new[act] - phip[act]).abs().max())
            phip = phip.clone(); phip[act] = new[act]
            if d < 1e-10: break
        return phip

    # ------------------------------------------------------------------ public
    def refine(self, guess, vg, vd) -> RefineResult:
        """guess: (n,3) psi, Phi_n, Phi_p [V] in DF-ISE vertex order (e.g. a GNN prediction)."""
        self.bk.activate()
        md, cfg = self.md, self.cfg
        t0 = time.monotonic()
        pd, phin, phip = self.boundary(guess, vg, vd)
        psi = np.asarray(guess, float)[:, 0].copy(); fin = np.isfinite(pd); psi[fin] = pd[fin]
        bounds = (min(0.0, vd) - cfg.phin_margin, max(0.0, vd) + cfg.phin_margin)
        semi = md.semi
        phin = np.where(semi & ~md.dirichlet, np.clip(phin, *bounds), phin)
        P, Fn, Fp = self.t(psi), self.t(np.where(semi, phin, 0.0)), self.t(np.where(semi, phip, 0.0))
        t1 = time.monotonic()
        P, Fn, gh = self.gummel(P, Fn, Fp, bounds)
        t2 = time.monotonic()
        nh_all, n_g = [], len(gh)
        budget = cfg.newton_maxit
        for k in range(cfg.restarts + 1):
            last = k == cfg.restarts
            P, Fn, nh = self.joint_newton(P, Fn, Fp, bounds, maxit=budget, stall_check=not last)
            nh_all += nh; budget -= max(len(nh) - 1, 0)
            if not nh or nh[-1][0] < cfg.newton_tol or budget <= 0:
                break
            P, Fn, g2 = self.gummel(P, Fn, Fp, bounds, iters=cfg.restart_gummel); n_g += len(g2)
        nh = nh_all
        t3 = time.monotonic()
        # hole start: Phi_p = Phi_n (p ~ n_i^2/n), with p limited to [~1e-100, ~1e10] cm^-3 so that the column
        # scaling of the linear hole solve stays representable (Phi_n is meaningless where n ~ 0)
        Ev = self.eq.C0 - self.eq.chi - P - self.eq.Eg
        Fh = self.holes(P, Fn, torch.clamp(Fn, -Ev - 276 * KT, -Ev - 21 * KT))
        t4 = time.monotonic()
        out_n = np.where(semi, Fn.cpu().numpy(), np.asarray(guess, float)[:, 1])
        if not (np.isfinite(P.cpu().numpy()).all() and np.isfinite(out_n).all() and bool(torch.isfinite(Fh).all())):
            raise FloatingPointError("refinement produced non-finite fields")
        # quasi-Fermi levels lie within the electrode-bias range; clamp as a last guard
        lo, hi = min(0.0, vg, vd) - 1.0, max(0.0, vg, vd) + 1.0
        out_p = np.where(semi, np.clip(Fh.cpu().numpy(), lo, hi), np.asarray(guess, float)[:, 2])
        m = nh[-1] if nh else (np.nan, np.nan, np.nan)
        return RefineResult(P.cpu().numpy(), out_n, out_p, bool(m[0] < 1e-6), n_g, len(nh),
                            float(m[0]), float(m[1]), float(m[2]),
                            dict(boundary=t1 - t0, gummel=t2 - t1, newton=t3 - t2, holes=t4 - t3, total=t4 - t0))

    def _contact_flow(self, psi, phin, phip):
        """Per-vertex conventional current [A] flowing from the semiconductor into the metal (SG residuals of
        the electron and hole equations at the metal-interface vertices)."""
        self.bk.activate()
        eq, md = self.eq, self.md
        with torch.no_grad():
            P, Fn, Fp = self.t(psi), self.t(np.where(md.semi, phin, 0)), self.t(np.where(md.semi, phip, 0))
            rn, _ = eq.electron(P, Fn)
            r, c, v, a, b, p = eq.hole_operator(P, Fn, Fp)
            A = self.bk.coo(r, c, v, (self.n, self.n))
            rp = self.bk.tens(A @ self.bk.arr(p)).to(DT)
        return Q * (rn.cpu().numpy() - rp.cpu().numpy())

    def metal_drop(self, psi, phin, phip):
        """Fermi-potential drop (phi_metal - V_contact) [V] at every contact-metal vertex, from the current the
        given fields carry into each contact: current conservation in the metal (box Laplacian over metal
        elements, conductance C / resistivity), phi = V on the contact sheet. Sentaurus keeps this metal state
        in QuasiFermiPotential; a warm-start guess with the metals pinned at the applied bias instead starts
        with a residual of ~1e9 in the metal equation, which Sentaurus' first Newton step does not survive at
        deep off-state (the guess then needs 55+ iterations or diverges; with the drop: 2)."""
        md = self.md
        if getattr(self, "_metal", None) is None:
            import scipy.sparse as sps
            import scipy.sparse.linalg as sla
            from scipy.sparse.csgraph import connected_components
            rho = getattr(md.physics, "metal_resistance", None) or METAL_RESISTANCE
            sel = np.isin(md.emat, list(rho))
            T, C = md.T[sel], md.C[sel]
            g = C / np.array([rho[m] for m in md.emat[sel]])[:, None]
            r, c, v = [], [], []
            for k, (a, b) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
                i, j, w = T[:, a], T[:, b], g[:, k]
                r += [i, j, i, j]; c += [j, i, i, j]; v += [-w, -w, w, w]
            L = sps.coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(md.n, md.n)).tocsr()
            self._metal = {}
            for e in ("S", "D", "G"):
                verts = np.where(md.electrode == e)[0]
                if e not in md.contact_sets or not len(verts):
                    continue
                fixed = np.zeros(md.n, bool); fixed[md.contact_sets[e]] = True
                free = verts[~fixed[verts]]
                if not len(free):
                    continue
                anchored = np.asarray(abs(L[free][:, np.where(fixed)[0]]).sum(1)).ravel() > 0
                _, lab = connected_components(L[free][:, free], directed=False)
                free = free[np.isin(lab, np.unique(lab[anchored]))]     # floating metal keeps the contact bias
                if len(free):
                    self._metal[e] = (free, sla.splu(L[free][:, free].tocsc()))
        flow = self._contact_flow(psi, phin, phip)
        drop = np.zeros(md.n)
        for free, lu in self._metal.values():
            drop[free] = lu.solve(np.where(md.semi[free], -flow[free], 0.0))
        return drop

    def terminal_currents(self, psi, phin, phip, area_factor=1.0):
        """Terminal currents from the SG fluxes: q x (electron - hole) particle flow from the semiconductor
        into each electrode's metal [A], times `area_factor` (with the deck's AreaFactor the result is in
        the units Sentaurus writes to the .plt, i.e. mA/mm for the palto decks). Positive = into the device
        terminal (drain current > 0 in on-state). Validated against Sentaurus on true solutions: ~5e-4 rel."""
        flow, md = self._contact_flow(psi, phin, phip), self.md
        # electrons flowing into the metal leave the device: conventional current into the terminal = +q*flow
        return {c: float(flow[(md.electrode == c) & md.semi].sum()) * area_factor for c in ("S", "D", "G")}
