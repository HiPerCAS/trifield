"""Exact discrete drift-diffusion equations of the production deck (torch, float64; CPU or GPU).

Unknowns per vertex: psi, Phi_n, Phi_p [V]. All residuals are assembled with Sentaurus' own box data
(`MeshData`) element by element:

Poisson (C):    r_i = sum_T sum_edges eps0 eps_T kappa_ab (psi_b - psi_a) + alpha q sum_T M_Ti SC_i
                SC = p - n + N + NT (single-valued per vertex: no double points in the deck)
Electrons (1/s): r_i = sum_T sum_edges mu_T V_T kappa_ab [n_a B(-d) - n_b B(d)] (inflow)  - M_i R_i
                d = (w_b - w_a)/V_T, w = psi + chi + V_T ln gamma_n  (Scharfetter-Gummel, Fermi-corrected)
                mu_T = measure-weighted mean over the element's vertices of Caughey-Thomas(mu_low,v, F_T),
                F_T = |grad Phi_n| (|grad psi| in elements touching a contact sheet)
Holes:          mirror image with w_p = psi + chi + E_g - V_T ln gamma_p, R = SRH + interface-trap capture.
Recombination is negligible for electrons (R <~ 1e2 cm^-3 s^-1) and kept only in the hole equation, where
it sets Phi_p (p ~ 1e-50).
"""
from __future__ import annotations

import numpy as np
import torch

from trifield.physics.fermi import KT, r_of_eta, r_of_eta_t, eta_of_r, gamma_of_eta
from trifield.physics.mesh_data import EDGES, MeshData

DT = torch.float64


def bern(x):
    small = x.abs() < 1e-4
    xs = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x / 2 + x * x / 12, xs / torch.expm1(xs))


class Equations:
    """Torch tensors of a MeshData on one device + residual / Jacobian evaluators."""

    def __init__(self, md: MeshData, device="cpu"):
        self.md = md
        self.dev = torch.device(device)
        t = lambda a, dt=DT: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)
        self.n = md.n
        self.chi, self.Eg, self.Nc, self.Nv, self.dop = t(md.chi), t(md.Eg), t(md.Nc), t(md.Nv), t(md.dop)
        self.semi = t(md.semi, torch.bool)
        self.C0 = md.C0
        sem = md.sem_e
        self.Ts = torch.as_tensor(md.T[sem], device=self.dev)
        self.Tsn = md.T[sem]
        self.Cs = t(md.C[sem]); self.Ms = t(md.M[sem])
        self.G = t(md.grad_op[sem]); self.tc = t(md.touch_contact_e[sem], torch.bool)
        self.MLn = t(md.mu_low_n[sem]); self.MLp = t(md.mu_low_p[sem])
        ph = md.physics
        self.bn, self.bp, self.vs = ph.beta_n, ph.beta_p, ph.vsat
        self.traps = [(torch.as_tensor(i, device=self.dev), c, e, cc, t(K)) for i, c, e, cc, K in md.traps]
        self.qM = t(md.qM)
        self.pdiag = t(md.pdiag)
        self.Msemi = t(md.Msemi * 1e-12)                 # cm^3 (recombination sink)

    # -------------------------------------------------------------- densities and charge
    def bands(self, psi, phin, phip):
        Ec = self.C0 - self.chi - psi
        Ev = Ec - self.Eg
        en = (-phin - Ec) / KT
        ep = (Ev + phip) / KT
        rn, rp = r_of_eta_t(en), r_of_eta_t(ep)
        gn = torch.where(en < -30, torch.ones_like(en), rn * torch.exp(-torch.clamp(en, min=-30)))
        gp = torch.where(ep < -30, torch.ones_like(ep), rp * torch.exp(-torch.clamp(ep, min=-30)))
        return Ec, Ev, self.Nc * rn, self.Nv * rp, gn, gp

    def trapped(self, Ec, Ev, n, p, gn, gp):
        NT = torch.zeros_like(n)
        for idx, conc, emid, cc, K in self.traps:
            Et = (Ec[idx] + Ev[idx]) / 2 + KT / 2 * torch.log(self.Nv[idx] / self.Nc[idx]) + emid
            n1 = gn[idx] * self.Nc[idx] * torch.exp((Et - Ec[idx]) / KT)
            p1 = gp[idx] * self.Nv[idx] * torch.exp((Ev[idx] - Et) / KT)
            sig = conc * (n1 + p[idx]) / (n[idx] + n1 + p[idx] + p1)          # ionised donors (c_n = c_p)
            NT = NT.index_add(0, idx, K * sig)
        return NT

    def space_charge(self, psi, phin, phip):
        Ec, Ev, n, p, gn, gp = self.bands(psi, phin, phip)
        sc = p - n + self.dop + self.trapped(Ec, Ev, n, p, gn, gp)
        return torch.where(self.semi, sc, torch.zeros_like(sc))

    # -------------------------------------------------------------- Poisson
    def poisson(self, psi, phin, phip, K):
        """Residual [C] (K: torch sparse flux operator)."""
        return K @ psi + self.qM * self.space_charge(psi, phin, phip)

    # -------------------------------------------------------------- mobility
    def mobility_e(self, P, F, holes=False):
        """Element mobility from element-local psi P (nE,4) and quasi-Fermi F (nE,4)."""
        gF = torch.einsum("eij,ej->ei", self.G, F[:, 1:] - F[:, :1])
        gP = torch.einsum("eij,ej->ei", self.G, P[:, 1:] - P[:, :1])
        Fd = torch.where(self.tc, gP.norm(dim=1), torch.sqrt((gF ** 2).sum(1) + 1e-18))
        ML, b = (self.MLp, self.bp) if holes else (self.MLn, self.bn)
        mu = ML / (1 + (ML * Fd[:, None] / self.vs) ** b) ** (1 / b)
        return (mu * self.Ms).sum(1) / torch.clamp(self.Ms.sum(1), min=1e-300)

    # -------------------------------------------------------------- electron continuity (element-local)
    def electron_elem(self, P, F):
        chi = self.chi[self.Ts]; Nc = self.Nc[self.Ts]
        eta = (-F - (self.C0 - chi - P)) / KT
        r = r_of_eta_t(eta)
        n = Nc * r
        w = P + chi + KT * (torch.log(r) - eta)
        mu = self.mobility_e(P, F)
        res = [torch.zeros_like(P[:, 0]) for _ in range(4)]; mag = [torch.zeros_like(P[:, 0]) for _ in range(4)]
        for k, (a, b) in enumerate(EDGES):
            d = (w[:, b] - w[:, a]) / KT
            f = mu * KT * self.Cs[:, k] * 1e-4 * (n[:, a] * bern(-d) - n[:, b] * bern(d))     # particles/s
            res[a] = res[a] - f; res[b] = res[b] + f
            mag[a] = mag[a] + f.abs(); mag[b] = mag[b] + f.abs()
        return torch.stack(res, 1), torch.stack(mag, 1)

    def electron(self, psi, phin):
        er, em = self.electron_elem(psi[self.Ts], phin[self.Ts])
        z = torch.zeros(self.n, dtype=DT, device=self.dev)
        return z.index_add(0, self.Ts.reshape(-1), er.reshape(-1)), z.index_add(0, self.Ts.reshape(-1), em.reshape(-1))

    def electron_jacobian(self, psi, phin):
        """Residual, magnitude and sparse (torch COO -> returned as numpy triplets) d r / d psi, d r / d Phi_n."""
        P = psi[self.Ts].detach().clone().requires_grad_(True)
        F = phin[self.Ts].detach().clone().requires_grad_(True)
        er, em = self.electron_elem(P, F)
        rows, cols, vp, vf = [], [], [], []
        for i in range(4):
            gp, gf = torch.autograd.grad(er[:, i].sum(), (P, F), retain_graph=True)
            for k in range(4):
                rows.append(self.Ts[:, i]); cols.append(self.Ts[:, k]); vp.append(gp[:, k]); vf.append(gf[:, k])
        z = torch.zeros(self.n, dtype=DT, device=self.dev)
        r = z.index_add(0, self.Ts.reshape(-1), er.detach().reshape(-1))
        m = z.index_add(0, self.Ts.reshape(-1), em.detach().reshape(-1))
        return r, m, torch.cat(rows), torch.cat(cols), torch.cat(vp), torch.cat(vf)

    def electron_sg_operator(self, psi, phin):
        """Linear SG operator A (inflow = A n) with w and mobility frozen; returns triplets and n."""
        with torch.no_grad():
            P, F = psi[self.Ts], phin[self.Ts]
            chi = self.chi[self.Ts]
            eta = (-F - (self.C0 - chi - P)) / KT
            r = r_of_eta_t(eta)
            w = P + chi + KT * (torch.log(r) - eta)
            mu = self.mobility_e(P, F)
            rows, cols, vals = [], [], []
            for k, (a, b) in enumerate(EDGES):
                d = (w[:, b] - w[:, a]) / KT; c = mu * KT * self.Cs[:, k] * 1e-4
                Bm, Bp = bern(-d), bern(d); ia, ib = self.Ts[:, a], self.Ts[:, b]
                rows += [ia, ia, ib, ib]; cols += [ia, ib, ia, ib]; vals += [-c * Bm, c * Bp, c * Bm, -c * Bp]
            n_full = torch.zeros(self.n, dtype=DT, device=self.dev)
            n_full[self.Ts.reshape(-1)] = (self.Nc[self.Ts] * r).reshape(-1)
        return torch.cat(rows), torch.cat(cols), torch.cat(vals), n_full

    # -------------------------------------------------------------- holes (linear in p, lagged)
    def hole_operator(self, psi, phin, phip):
        """Inflow = A p (SG) and recombination R = a p - b (lagged denominators); numpy-free triplets."""
        md = self.md
        with torch.no_grad():
            Ec, Ev, n, p, gn, gp = self.bands(psi, phin, phip)
            w = psi + self.chi + self.Eg - KT * torch.log(gp)
            mu = self.mobility_e(psi[self.Ts], phip[self.Ts], holes=True)
            rows, cols, vals = [], [], []
            for k, (a_, b_) in enumerate(EDGES):
                ia, ib = self.Ts[:, a_], self.Ts[:, b_]
                d = (w[ib] - w[ia]) / KT; c = mu * KT * self.Cs[:, k] * 1e-4
                Bp, Bm = bern(d), bern(-d)
                rows += [ia, ia, ib, ib]; cols += [ia, ib, ia, ib]; vals += [-c * Bp, c * Bm, c * Bp, -c * Bm]
            nie2 = self.Nc * self.Nv * torch.exp(-self.Eg / KT); nie = torch.sqrt(nie2)
            tau = md.physics.tau
            den = tau * (n + gn * nie) + tau * (p + gp * nie)
            a = n / den; b = gn * gp * nie2 / den
            for idx, conc, emid, cc, K in self.traps:
                Et = (Ec[idx] + Ev[idx]) / 2 + KT / 2 * torch.log(self.Nv[idx] / self.Nc[idx]) + emid
                n1 = gn[idx] * self.Nc[idx] * torch.exp((Et - Ec[idx]) / KT)
                p1 = gp[idx] * self.Nv[idx] * torch.exp((Ev[idx] - Et) / KT)
                d2 = n[idx] + n1 + p[idx] + p1
                a = a.index_add(0, idx, K * conc * cc * n[idx] / d2)
                b = b.index_add(0, idx, K * conc * cc * n1 * p1 / d2)
        return torch.cat(rows), torch.cat(cols), torch.cat(vals), a, b, p

    def phip_from_p(self, psi, p):
        Ev = self.C0 - self.chi - psi - self.Eg
        eta = torch.as_tensor(eta_of_r(np.maximum((p / self.Nv).cpu().numpy(), 1e-300)), dtype=DT, device=self.dev)
        return KT * eta - Ev

    def phin_from_n(self, psi, n):
        Ec = self.C0 - self.chi - psi
        eta = torch.as_tensor(eta_of_r(np.maximum((n / self.Nc).cpu().numpy(), 1e-300)), dtype=DT, device=self.dev)
        return -(Ec + KT * eta)
