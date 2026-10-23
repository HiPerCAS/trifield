"""Design space of the multi-fin tri-gate GaN FinFET test case.

``PARAM_ORDER`` fixes the ordering used whenever a design is converted to or
from a vector. Keep it stable: the acquisition diversity filter and any
normalisation statistics depend on it.
"""
from __future__ import annotations

import numpy as np

# The six continuous design variables (units: um, except doping in cm^-3).
PARAM_ORDER: list[str] = [
    "GaN",           # GaN channel thickness        T_GaN   [um]
    "AlGaN",         # AlGaN barrier thickness      T_AlGaN [um]
    "Fw",            # fin width                    W_fin   [um]
    "Tox",           # gate-oxide thickness         T_ox    [um]
    "Hg",            # gate metal thickness         T_Gate  [um]
    "AlGaN_doping",  # barrier doping               N_AlGaN [cm^-3]
]

# Exploration ranges (Table 1 of the paper).
RANGES: dict[str, tuple[float, float]] = {
    "GaN":          (0.01, 0.10),
    "AlGaN":        (0.01, 0.10),
    "Fw":           (0.08, 0.25),
    "Tox":          (0.005, 0.025),
    "Hg":           (0.005, 0.040),
    "AlGaN_doping": (1e19, 1e20),
}

# Parameters held constant across the campaign.
FIXED: dict[str, float] = {
    "ch_doping": 1e14,   # GaN channel doping [cm^-3]
    "Vg11": 5.0,         # maximum gate bias V_GS,max [V]
    "Lext2": 0.6,        # drain-side spacer length [um]
}

# Inter-fin spacing [um]; must match ``(define Sfin 0.1)`` in the SDE template.
SFIN: float = 0.1

# The nine bias snapshots written per Sentaurus run: (tag, Vg / Vg11, Vd [V]).
# Five gate-sweep points at Vd = 0 and four drain-sweep points at Vg = Vg11.
BIAS_SNAPSHOTS: list[tuple[str, float, float]] = [
    ("snap_vg02_vd00", 0.2, 0.0),
    ("snap_vg04_vd00", 0.4, 0.0),
    ("snap_vg06_vd00", 0.6, 0.0),
    ("snap_vg08_vd00", 0.8, 0.0),
    ("snap_vg10_vd00", 1.0, 0.0),
    ("snap_vg10_vd03", 1.0, 3.0),
    ("snap_vg10_vd06", 1.0, 6.0),
    ("snap_vg10_vd09", 1.0, 9.0),
    ("node1_des",      1.0, 12.0),   # final Vd = 12 V corner
]


def snapshot_stem(tag: str) -> str:
    """Graph / TDR file stem of a bias snapshot (``<design>__<stem>.pt``)."""
    return tag if tag == "node1_des" else f"{tag}_des"


def fin_pitch(params: dict) -> float:
    """Fin pitch W_fin + S_fin [um] (the y-normalisation unit of the graph)."""
    return float(params["Fw"]) + SFIN


def area_factor(params: dict) -> float:
    """Sentaurus ``AreaFactor`` so that currents are per mm of total device
    width: 1e6 / (N_fins (perimeter + W_fin + S_fin)), perimeter = 8 (T_GaN + T_AlGaN)."""
    perimeter = 8.0 * (float(params["GaN"]) + float(params["AlGaN"]))
    nfins = int(params.get("Nfins", 1))
    return 1e6 / (nfins * (perimeter + float(params["Fw"]) + SFIN))


def is_feasible(params: dict, strict: bool = False) -> bool:
    """Geometric feasibility of a multi-fin design.

    The inter-fin gap must hold the gate stack of both fins:
    ``T_ox + T_Gate < S_fin / 2``. With ``strict=True`` the convergence-yield
    filter used for the training campaign is applied as well (wide-fin,
    thin-GaN and thick-oxide corners fail to converge for N_fins > 1).
    """
    nfins = int(params.get("Nfins", 1))
    if nfins > 1 and (float(params["Tox"]) + float(params["Hg"])) >= SFIN / 2:
        return False
    if strict and nfins > 1 and (float(params["Fw"]) > 0.18
                                 or float(params["GaN"]) < 0.025
                                 or float(params["Tox"]) > 0.022):
        return False
    return True


def sample_design(rng: np.random.Generator, nfins: int = 1,
                  strict: bool = False, max_tries: int = 100) -> dict | None:
    """Draw one feasible design uniformly over ``RANGES`` (plus ``FIXED``)."""
    for _ in range(max_tries):
        p = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in RANGES.items()}
        p.update(FIXED)
        p["Nfins"] = int(nfins)
        if is_feasible(p, strict=strict):
            return p
    return None


def design_vector(params: dict) -> np.ndarray:
    """Map the continuous design variables to [0, 1]^6 by their ranges."""
    out = []
    for k in PARAM_ORDER:
        lo, hi = RANGES[k]
        out.append((float(params[k]) - lo) / max(hi - lo, 1e-30))
    return np.asarray(out, dtype=np.float64)


def snapshot_biases(params: dict) -> list[tuple[str, float, float]]:
    """The nine (tag, Vg, Vd) bias points of a design, in volts."""
    vg11 = float(params.get("Vg11", FIXED["Vg11"]))
    return [(tag, frac * vg11, vd) for tag, frac, vd in BIAS_SNAPSHOTS]
