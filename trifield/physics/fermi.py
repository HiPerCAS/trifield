"""Sentaurus' Fermi-Dirac statistics (numpy + torch).

Sentaurus does not evaluate the Fermi integral F_1/2 exactly. Densities follow from inverting its
approximation of eta(r), r = n / N_C:
    r <= 8.463 : eta = ln r + sum_k A_k r^k                (Joyce-Dixon, 4 terms)
    r >  8.463 : eta = sqrt((3 sqrt(pi) r / 4)^(4/3) - pi^2/6) - c,  c = continuity at r = 8.463
A1..A3 are the exact Joyce-Dixon values, A4 was fitted to exported densities (residual 2.6e-13);
the exact F_1/2 differs from Sentaurus by up to 2e-3 for eta > 2.
kT was fitted to exported hole densities (ln-residual 5e-13).
"""
from __future__ import annotations

import numpy as np

KT = 0.02585199526648491                       # eV (Sentaurus' value at 300 K)
JD = (0.35355339059327373, 3 / 16 - 3 ** 0.5 / 9, 1 / 8 + 5 * 2 ** 0.5 / 48 - 6 ** 0.5 / 9, -4.425630118993284e-06)
R0 = 8.463


def eta_jd(r):
    return np.log(r) + sum(a * r ** (k + 1) for k, a in enumerate(JD))


def eta_asy(r):
    return np.sqrt((3 * np.sqrt(np.pi) * r / 4) ** (4 / 3) - np.pi ** 2 / 6)


ETA0 = float(eta_jd(R0))
C_ASY = float(eta_asy(R0) - ETA0)


def eta_of_r(r):
    """Inverse statistics: eta for a density ratio r = n/N_C (numpy)."""
    r = np.asarray(r, float)
    return np.where(r <= R0, eta_jd(r), eta_asy(np.maximum(r, R0)) - C_ASY)


def r_of_eta(eta, iters=40):
    """n / N_C as a function of eta (numpy)."""
    eta = np.asarray(eta, float)
    e = np.maximum(eta, 0.0)
    asy = 4 / (3 * np.sqrt(np.pi)) * ((e + C_ASY) ** 2 + np.pi ** 2 / 6) ** 0.75
    lr = np.where(eta < 0, eta, np.log(asy))
    en = np.minimum(eta, ETA0)
    for _ in range(iters):
        r = np.exp(lr)
        f = lr + sum(a * r ** (k + 1) for k, a in enumerate(JD)) - en
        lr = lr - f / (1 + sum((k + 1) * a * r ** (k + 1) for k, a in enumerate(JD)))
    return np.where(eta > ETA0, asy, np.exp(lr))


def gamma_of_eta(eta, r=None):
    """Degeneracy factor gamma = r / exp(eta) (1 in the Boltzmann limit)."""
    r = r_of_eta(eta) if r is None else r
    return np.where(eta < -30, 1.0, r * np.exp(-np.maximum(eta, -30)))


# ------------------------------------------------------------------ torch
def r_of_eta_t(eta, iters=30):
    """Differentiable n/N_C(eta): Newton without graph, then one differentiable step whose gradient is
    the exact implicit-function derivative."""
    import torch
    jd = torch.tensor(JD, dtype=eta.dtype, device=eta.device)
    kk = torch.arange(1, 5, dtype=eta.dtype, device=eta.device)
    deg = eta > ETA0
    e = torch.clamp(eta, min=0.0)
    asy = 4 / (3 * np.sqrt(np.pi)) * ((e + C_ASY) ** 2 + np.pi ** 2 / 6) ** 0.75
    en = torch.clamp(eta, max=ETA0)
    with torch.no_grad():
        lr = torch.where(eta < 0, eta, torch.log(asy))
        for _ in range(iters):
            pw = torch.exp(lr)[..., None] ** kk
            lr = lr - (lr + (pw * jd).sum(-1) - en) / (1 + (pw * jd * kk).sum(-1))
    pw = torch.exp(lr)[..., None] ** kk
    f = lr + (pw * jd).sum(-1) - en
    df = 1 + (pw * jd * kk).sum(-1)
    lr = lr - f / df.detach()
    return torch.where(deg, asy, torch.exp(lr))
