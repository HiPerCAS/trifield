"""Per-mesh data for the exact Sentaurus residuals: geometry, box-method data, material constants.

Everything here is bias independent and is built once per design from three Sentaurus outputs
(see ``trifield.physics.export``):

1. the TDR grid the device is simulated on (element connectivity in Sentaurus' local order),
2. ``MeasureCoefficientsDebug`` written in Sentaurus' internal numbering (box measures and
   coefficients; the GrdNumbering variant corrupts the coefficients: it writes internal slots
   [0,3,1,3,2,4], i.e. edge (2,3) is lost and slot 3 duplicated),
3. a DF-ISE Plot of an equilibrium solution with the band/density parameters.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trifield.physics.tdr import tetrahedra

EPS0 = 8.8541878128e-14          # F/cm
Q = 1.602176634e-19              # C
METAL_RESISTANCE = {"Gold": 2.2603e-2, "TiN": 2.2603e-2}    # see DeckPhysics.metal_resistance (TiN: no gate current)
ALPHA = 1.0000082142581186       # Sentaurus' eps0/q ratio vs CODATA (fitted, interior Poisson -> 1e-14)
EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))   # manual Fig. 101 = internal coefficient order
SEMI = ("GaN", "AlGaN")
INSULATORS = ("HfO2",)
METALS = ("Gold", "TiN", "Aluminum", "Copper", "Metal", "PolySi")


@dataclass
class TrapSet:
    """Interface donor trap level (Physics(MaterialInterface=...){Traps(...)})."""
    materials: tuple            # sorted material pair, e.g. ("AlGaN", "GaN")
    conc: float                 # cm^-2
    e_mid: float                # eV above E_i (FromMidBandGap)
    c_capture: float            # vth * sigma [cm^3/s] (equal for e and h)


@dataclass
class DeckPhysics:
    """Physics of the production deck + parameter.par that the residuals need (defaults: palto deck)."""
    eps: dict = field(default_factory=lambda: {"GaN": 9.5, "HfO2": 22.0, "AlGaN": (8.9, 8.5)})   # AlGaN: (x=0, x=1)
    # low-field mobility: doping dependence is inactive (Sentaurus sees N=0) -> mu = mumax(x) - mumin2
    mu_n: dict = field(default_factory=lambda: {"GaN": 1700.0 - 75.0, "AlGaN": ((1500.0, 300.0), 75.0)})
    mu_p: dict = field(default_factory=lambda: {"GaN": 20.0, "AlGaN": ((150.0, 14.0), 0.0)})
    beta_n: float = 0.9
    beta_p: float = 0.8
    vsat: float = 2.6e7
    tau: float = 1e-9                           # SRH lifetimes (taumax, N = 0)
    traps: tuple = (TrapSet(("AlGaN", "GaN"), 3e13, 1.25, 1.5626e7 * 1e-14),
                    TrapSet(("AlGaN", "HfO2"), 5e13, 0.6, 1.5626e7 * 1e-15))
    # contact metals conduct: their Fermi potential drops by up to ~1.5 mV between the contact sheet and the
    # semiconductor interface. Value = resistivity in box-coefficient units (drop = L^-1 I with L built from
    # C / value); calibrated on two designs (2.2603e-2, 2.2651e-2), reproduces Sentaurus to ~0.3%.
    metal_resistance: dict = field(default_factory=lambda: dict(METAL_RESISTANCE))


def _tri_voronoi(P):
    a = np.zeros(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        cot = lambda o, p, q: np.dot(P[p] - P[o], P[q] - P[o]) / np.linalg.norm(np.cross(P[p] - P[o], P[q] - P[o]))
        a[i] = (np.sum((P[i] - P[j]) ** 2) * cot(k, i, j) + np.sum((P[i] - P[k]) ** 2) * cot(j, i, k)) / 8
    return a


def read_internal_box(path):
    """MeasureCoefficientsDebug (internal numbering): Measure (nE,4), Coefficient (nE,6), internal units."""
    txt = Path(path).read_text()
    m_txt, c_txt = txt.split("Coefficient {")
    M = np.array([l.split() for l in m_txt.split("{", 1)[1].split("}")[0].strip().splitlines()], float)
    C = np.array([l.split() for l in c_txt.split("}")[0].strip().splitlines()], float)
    return M, C


class MeshData:
    """All bias-independent arrays. Vertex indices follow the DF-ISE grid (= GNN graph node order)."""

    def __init__(self, grid_tdr, dfise_grd, box_debug, const_dat, physics: DeckPhysics | None = None):
        from trifield.data.tdr_extract import parse_dat, parse_grd
        self.physics = ph = physics or DeckPhysics()
        mesh = parse_grd(dfise_grd)
        self.mesh = mesh
        self.X = np.asarray(mesh.vertices, float)          # um
        self.n = n = mesh.n_vertices
        T, emat, ereg = tetrahedra(grid_tdr)
        self.T, self.emat, self.ereg = T, emat, ereg
        Mi, Ci = read_internal_box(box_debug)
        if len(Mi) != len(T):
            raise ValueError(f"box export has {len(Mi)} elements, TDR grid {len(T)} (des numbering mismatch)")
        vol = np.abs(np.einsum("ij,ij->i", np.cross(self.X[T[:, 1]] - self.X[T[:, 0]], self.X[T[:, 2]] - self.X[T[:, 0]]),
                               self.X[T[:, 3]] - self.X[T[:, 0]])) / 6
        s = (vol.sum() / Mi.sum()) ** (1 / 3)                     # internal length unit -> um
        self.M, self.C = Mi * s ** 3, Ci * s                       # um^3, um
        # mapping check: Sentaurus conserves the total box volume per region (Delaunay-ish meshes)
        self.box_check = {r: float(abs(self.M[ereg == r].sum() / vol[ereg == r].sum() - 1)) for r in np.unique(ereg)}
        self.length_unit = s
        mat = dict(zip(mesh.region_names, mesh.region_materials)); self.mat = mat
        rvi = mesh.region_vertex_indices; self.rvi = rvi
        ds = parse_dat(const_dat)

        def vf(name, mats=SEMI):
            v = np.full(n, np.nan)
            for d in ds:
                if d.name == name and d.region in rvi and "+" not in d.region and d.values.ndim == 1 \
                        and len(d.values) == len(rvi[d.region]) and (mats is None or mat[d.region] in mats):
                    v[rvi[d.region]] = d.values
            return v
        chi, Eg = vf("ElectronAffinity"), vf("EffectiveBandGap")
        self.semi = np.isfinite(chi)
        fill = lambda a, x: np.where(np.isfinite(a), a, x)
        self.chi, self.Eg = fill(chi, 4.0), fill(Eg, 3.4)
        self.Nc, self.Nv = fill(vf("eEffectiveStateDensity"), 1e18), fill(vf("hEffectiveStateDensity"), 1e18)
        self.dop = fill(vf("DopingConcentration"), 0.0)
        psi_eq = np.nan_to_num(vf("ElectrostaticPotential", None))
        self.C0 = float(np.nanmedian(vf("ConductionBandEnergy") + vf("ElectrostaticPotential") + chi))
        xm = vf("xMoleFraction", ("AlGaN",))
        self.x_algan = float(np.nanmedian(xm[np.isfinite(xm) & (xm > 0)])) if np.any(np.isfinite(xm) & (xm > 0)) else 0.0
        # vertex AlGaN parameter class: Sentaurus uses AlGaN(x of the vertex); x = 0 vertices have chi != GaN chi
        self.x0 = self.semi & np.isfinite(xm) & (xm == 0)
        # element permittivity
        epsA = ph.eps["AlGaN"][0] + self.x_algan * (ph.eps["AlGaN"][1] - ph.eps["AlGaN"][0])
        epsd = {k: v for k, v in ph.eps.items() if k != "AlGaN"}; epsd["AlGaN"] = epsA
        self.diel = np.isin(emat, list(epsd))
        self.eps_e = np.array([epsd.get(m, 0.0) for m in emat])
        self.sem_e = np.isin(emat, SEMI)
        # vertex classes
        touch = collections.defaultdict(set)
        for e, t in enumerate(T):
            for v in t: touch[int(v)].add(emat[e])
        self.metal_v = np.array([bool(touch[v] & set(METALS)) for v in range(n)])
        self.metal_only = np.array([bool(touch[v]) and touch[v] <= set(METALS) for v in range(n)])
        contact = np.zeros(n, bool)
        self.contact_sets = {}
        for c in ("S", "D", "G"):
            if c in rvi:
                contact[rvi[c]] = True; self.contact_sets[c] = np.asarray(rvi[c])
        self.dirichlet = self.metal_v | contact
        self.in_eq = np.zeros(n, bool); self.in_eq[np.unique(T[self.diel])] = True
        self.cls = np.empty(n, object)
        for v in range(n):
            ms = touch[v] - set(METALS)
            self.cls[v] = ("dirichlet" if self.dirichlet[v] else "hetero:" + "/".join(sorted(ms)) if len(ms) > 1
                           else "interior:" + next(iter(ms)) if ms else "none")
        self.cls = self.cls.astype(str)
        # electrodes: metal regions -> contact they touch; vertex -> electrode name ('' if none)
        self.electrode = np.full(n, "", dtype=object)
        self.metal_material = np.full(n, "", dtype=object)
        for reg, m in zip(mesh.region_names, mesh.region_materials):
            if m in METALS and reg in rvi:
                idx = rvi[reg]; vs = set(map(int, idx))
                hits = {c: len(vs & set(map(int, cv))) for c, cv in self.contact_sets.items()}
                if hits and max(hits.values()) > 0:
                    self.electrode[idx] = max(hits, key=hits.get)
                self.metal_material[idx] = m
        # boundary constants from the equilibrium export (Phi = 0 there)
        gate = (self.metal_material == "TiN")
        self.gate_offset = float(np.median(psi_eq[gate])) if gate.any() else 0.846115
        gold_only = self.metal_only & (self.metal_material == "Gold")
        self.gold_offset = float(np.median(psi_eq[gold_only])) if gold_only.any() else 0.5061
        x0d = self.dirichlet & self.x0
        self.x0_offset = float(np.median(psi_eq[x0d])) if x0d.any() else -0.140375
        # box volumes, interface areas, traps
        self.Msemi = np.zeros(n)
        for j in range(4): np.add.at(self.Msemi, T[self.sem_e, j], self.M[self.sem_e, j])
        pair_area = collections.defaultdict(lambda: np.zeros(n))
        for rn, els in mesh.region_element_indices.items():
            if "+" not in rn: continue
            pr = tuple(sorted(mat[q] for q in rn.split("+")))
            for e in els:
                tri = mesh.all_elements[int(e)]
                if len(tri) == 3:
                    np.add.at(pair_area[pr], tri, _tri_voronoi(self.X[tri]))
        self.traps = []
        for ts in ph.traps:
            A = pair_area.get(tuple(sorted(ts.materials)))
            if A is None: continue
            idx = np.where((A > 0) & self.semi & (self.Msemi > 0))[0]
            self.traps.append((idx, ts.conc, ts.e_mid, ts.c_capture, A[idx] * 1e-8 / (self.Msemi[idx] * 1e-12)))
        # potential-equivalent normalisation for the Poisson residual
        dg = np.zeros(n)
        for k, (a, b) in enumerate(EDGES):
            w = np.abs(EPS0 * self.eps_e * self.C[:, k] * 1e-4); np.add.at(dg, T[:, a], w); np.add.at(dg, T[:, b], w)
        self.pdiag = np.where(dg > 0, dg, 1.0)
        self.qM = ALPHA * Q * self.Msemi * 1e-12
        # mobility per element-vertex
        mA_n = ph.mu_n["AlGaN"]; mA_p = ph.mu_p["AlGaN"]
        x_v = np.where(self.x0, 0.0, self.x_algan)
        mun_A = mA_n[0][0] + x_v * (mA_n[0][1] - mA_n[0][0]) - mA_n[1]
        mup_A = mA_p[0][0] + x_v * (mA_p[0][1] - mA_p[0][0]) - mA_p[1]
        isA = (emat == "AlGaN")[:, None]
        self.mu_low_n = np.where(isA, mun_A[T], ph.mu_n["GaN"])
        self.mu_low_p = np.where(isA, mup_A[T], ph.mu_p["GaN"])
        cv = np.zeros(n, bool)
        for v in self.contact_sets.values(): cv[v] = True
        self.touch_contact_e = cv[T].any(1)
        X = self.X[T] * 1e-4
        with np.errstate(all="ignore"):
            self.grad_op = np.linalg.inv(X[:, 1:] - X[:, :1])      # cm^-1
        self._K = None

    # ------------------------------------------------------------------ helpers
    @property
    def K(self):
        """Sparse Poisson flux operator (Coulomb/V): (K psi)_i = sum eps0 eps kappa (psi_j - psi_i)."""
        if self._K is None:
            import scipy.sparse as sps
            rows, cols, vals = [], [], []
            for k, (a, b) in enumerate(EDGES):
                w = EPS0 * self.eps_e * self.C[:, k] * 1e-4; ia, ib = self.T[:, a], self.T[:, b]
                rows += [ia, ia, ib, ib]; cols += [ib, ia, ib, ia]; vals += [w, -w, -w, w]
            self._K = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))
        return self._K

    def electrode_bias(self, vg, vd, vs=0.0):
        """Per-vertex electrode Fermi level (applied bias of the metal a vertex belongs to, else 0)."""
        b = {"S": vs, "D": vd, "G": vg}
        return np.array([b.get(e, 0.0) for e in self.electrode], dtype=float)

    def summary(self):
        return dict(n_vertices=self.n, n_tets=len(self.T), x_algan=self.x_algan, C0=self.C0,
                    gate_offset=self.gate_offset, gold_offset=self.gold_offset, x0_offset=self.x0_offset,
                    n_x0=int(self.x0.sum()), n_dirichlet=int(self.dirichlet.sum()),
                    traps=[(len(t[0]), t[1], t[2]) for t in self.traps],
                    box_volume_check_max=max(self.box_check.values()), length_unit=self.length_unit)
