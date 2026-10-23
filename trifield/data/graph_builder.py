"""Convert an extracted Sentaurus simulation into a PyG graph.

Node features x (N, 17):
    0-2   per-axis normalised coordinates (x / L_ch, y / fin pitch, z / z-range)
    3-7   material one-hot over MATERIALS = [GaN, AlGaN, Gold, HfO2, TiN]
    8     signed-log net doping  sign(N) log10(|N| + 1)
    9     Al mole fraction
    10-12 contact flags (source, drain, gate)
    13-14 applied bias (Vg, Vd) broadcast to all nodes
    15    relative permittivity (material lookup)
    16    log10 of the sparse Sentaurus DelVorWeight field (compatibility only)
Edge features edge_attr (E, 5): distance [um], displacement (3), cross-material flag.
Targets y (N, 3): (ElectrostaticPotential, eQuasiFermiPotential, hQuasiFermiPotential) [V].
Edges: the 6 vertex pairs of every tetrahedron, de-duplicated, stored in both directions.

Coordinate note: the SDE template defines x as the channel direction and z as
the growth direction, but the TDR -> DF-ISE conversion (``tdx --traditional``)
stores the mesh with x = layer stack, y = fin axis, z = channel (source at
z = 0). The normalisation below keeps the convention the released models were
trained with (column 0 divided by the channel length, column 1 by the fin
pitch, column 2 by its own range); it is applied identically at training and
inference time, so it is harmless.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional

import numpy as np
import torch

from trifield.data.tdr_extract import DFISEMesh, ExtractedSimulation

MATERIALS = ["GaN", "AlGaN", "Gold", "HfO2", "TiN"]
MATERIAL_TO_IDX = {m: i for i, m in enumerate(MATERIALS)}
MATERIAL_PERMITTIVITY = {"GaN": 9.0, "AlGaN": 9.4, "Gold": 1.0, "HfO2": 25.0, "TiN": 1.0}   # v1/v2 table
# relative permittivities Sentaurus applies per element (parameter file / exported box data)
MATERIAL_PERMITTIVITY_V3 = {"GaN": 9.5, "AlGaN": 8.8, "Gold": 1.0, "HfO2": 22.0, "TiN": 1.0}
CONTACT_FLAG_MAP = {"S": 0, "D": 1, "G": 2}

# Feature-version marker stored on every built graph.
#   1 (graphs built before 2026-07-18, no attribute): contact flags came from
#     a DF-ISE element-parsing bug (triangle references resolved as faces) and
#     marked the wrong node sets. The released checkpoints and example graphs
#     are version 1 and are consistent with each other.
#   2: contact flags from the contact regions' triangle elements (correct).
# Do not mix versions: features, normalisation statistics and trained weights differ.
#   3: material-dependent node features are exact at interfaces (about half of all
#      vertices: the GaN/AlGaN layers and the oxide are one element thick):
#      columns 3-7 = volume share of each material around the vertex (not a single
#      "last element wins" label), 8/9 = region-volume-weighted doping / Al fraction,
#      15 = volume-weighted permittivity (Sentaurus' values), edge column 4 = edge
#      shared by elements of 2+ materials; plus data.y_valid (quasi-Fermi targets are
#      simulator placeholders where no semiconductor is touched) and data.semi_any.
#      Ablated (35 epochs, same recipe): best v3 variant psi 0.270 V vs 0.253 V for v2 on
#      validation, so v2 stays the default; v3 remains available via feature_version=3.
FEATURE_VERSION = 2

LS = LEXT1 = LG = LD = 0.1   # source / source spacer / gate / drain lengths [um]


def _edges_from_tets(elements: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a tetrahedral mesh as a bidirectional (2, 2E) index."""
    tets = np.asarray(elements, dtype=np.int64)
    pairs = np.concatenate([tets[:, [i, j]] for i, j in combinations(range(4), 2)], axis=0)
    pairs.sort(axis=1)
    edges = np.unique(pairs, axis=0)
    return np.concatenate([edges, edges[:, ::-1]], axis=0).T


def _assign_vertex_materials(mesh: DFISEMesh) -> np.ndarray:
    """Material index per vertex from the regions of its incident tetrahedra (-1 = none)."""
    vertex_mat = np.full(mesh.n_vertices, -1, dtype=np.int32)
    for tet_idx, rid in enumerate(mesh.element_region_ids):
        if rid < 0:
            continue
        mat_idx = MATERIAL_TO_IDX.get(mesh.region_materials[rid], -1)
        if mat_idx >= 0:
            vertex_mat[mesh.elements[tet_idx]] = mat_idx
    return vertex_mat


def _vertex_region_volumes(mesh: DFISEMesh):
    """{region_id: (N,) incident tetrahedron volume share per vertex} (unnormalised)."""
    v = mesh.vertices[mesh.elements]
    vol = np.abs(np.einsum("ij,ij->i", v[:, 1] - v[:, 0],
                           np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]))) / 6.0
    out = {}
    for rid in np.unique(mesh.element_region_ids):
        if rid < 0:
            continue
        sel = mesh.element_region_ids == rid
        a = np.zeros(mesh.n_vertices)
        np.add.at(a, mesh.elements[sel].ravel(), np.repeat(vol[sel] / 4.0, 4))
        out[int(rid)] = a
    return out


def _interface_edges(mesh: DFISEMesh, edges: np.ndarray) -> np.ndarray:
    """True for (undirected, sorted) edges shared by tetrahedra of 2+ materials."""
    tets = np.sort(np.asarray(mesh.elements, dtype=np.int64), axis=1)
    mat = np.array([MATERIAL_TO_IDX.get(mesh.region_materials[r], -1) for r in mesh.element_region_ids])
    pairs = np.concatenate([tets[:, [i, j]] for i, j in combinations(range(4), 2)])
    pm = np.tile(mat, 6)
    key = pairs[:, 0] * mesh.n_vertices + pairs[:, 1]
    ek = edges[:, 0] * mesh.n_vertices + edges[:, 1]
    order = np.argsort(key)
    key, pm = key[order], pm[order]
    lo = np.searchsorted(key, ek, "left")
    hi = np.searchsorted(key, ek, "right")
    found = hi > lo
    lo_c = np.minimum(lo, len(key) - 1)
    mn = np.minimum.reduceat(pm, lo_c)
    mx = np.maximum.reduceat(pm, lo_c)
    return (mx != mn) & found


def build_graph(sim: ExtractedSimulation, vg: float = 0.0, vd: float = 0.0,
                params: Optional[Dict] = None, fin_pitch: Optional[float] = None,
                feature_version: int = FEATURE_VERSION):
    """Build the PyG ``Data`` for one bias snapshot of one design.

    ``params`` (design dict incl. ``Lext2`` and ``Nfins``) and ``fin_pitch``
    (``Fw + SFIN``) make the x / y normalisation deterministic across designs,
    which is what lets the model transfer across fin counts. Solution fields
    missing from ``sim`` (mesh-only candidates) give zero targets.
    """
    import torch_geometric.data as pyg_data

    mesh, mf, sf = sim.mesh, sim.mesh_fields, sim.solution_fields
    coords = mesh.vertices
    n = mesh.n_vertices

    edge_index = _edges_from_tets(mesh.elements)
    vertex_mat = _assign_vertex_materials(mesh)

    # 1. normalised coordinates
    lch = (LS + LEXT1 + LG + float(params["Lext2"]) + LD if params is not None
           else float(coords[:, 0].max() - coords[:, 0].min()))
    y_scale = (float(fin_pitch) if fin_pitch is not None
               else float(coords[:, 1].max() - coords[:, 1].min()))
    z_scale = float(coords[:, 2].max() - coords[:, 2].min())
    lch, y_scale, z_scale = (s if s else 1.0 for s in (lch, y_scale, z_scale))
    coords_norm = np.zeros_like(coords, dtype=np.float32)
    coords_norm[:, 0] = (coords[:, 0] - coords[:, 0].min()) / lch
    coords_norm[:, 1] = (coords[:, 1] - coords[:, 1].min()) / y_scale
    coords_norm[:, 2] = (coords[:, 2] - coords[:, 2].min()) / z_scale

    # 2. material encoding, 3. doping, 4. Al mole fraction
    doping = np.nan_to_num(mf.get("DopingConcentration", np.zeros(n)), nan=0.0)
    xmole = np.nan_to_num(mf.get("xMoleFraction", np.zeros(n)), nan=0.0)
    if feature_version >= 3:
        rvol = _vertex_region_volumes(mesh)
        mat_frac = np.zeros((n, len(MATERIALS)))
        for rid, a in rvol.items():
            mi = MATERIAL_TO_IDX.get(mesh.region_materials[rid], -1)
            if mi >= 0:
                mat_frac[:, mi] += a
        tot = mat_frac.sum(1, keepdims=True)
        mat_frac = np.divide(mat_frac, tot, out=np.zeros_like(mat_frac), where=tot > 0)
        mat_enc = mat_frac.astype(np.float32)
        doping_slog = None
        rf = getattr(mesh, "region_fields", {}) or {}
        slog = lambda v: np.sign(v) * np.log10(np.abs(v) + 1.0)
        doping_slog = None
        for name in ("DopingConcentration", "xMoleFraction"):
            per_reg = rf.get(name)
            if not per_reg:
                continue
            acc = np.zeros(n); wsum = np.zeros(n)
            for rid, a in rvol.items():
                reg = mesh.region_names[rid]
                if reg in per_reg:
                    vals = np.zeros(n); idx = mesh.region_vertex_indices[reg]
                    # doping is averaged in signed-log space: a raw average lets a
                    # sliver of 7e19 cm^-3 make a mostly-undoped vertex look degenerate
                    vals[idx] = slog(per_reg[reg]) if name == "DopingConcentration" else per_reg[reg]
                    acc += a * vals; wsum += a
            ok = wsum > 0
            avg = acc / np.where(ok, wsum, 1.0)
            if name == "DopingConcentration":
                doping_slog = np.where(ok, avg, slog(doping))
            else:
                xmole = np.where(ok, avg, xmole)
    else:
        mat_enc = np.zeros((n, len(MATERIALS)), dtype=np.float32)
        has_mat = vertex_mat >= 0
        mat_enc[np.where(has_mat)[0], vertex_mat[has_mat]] = 1.0
    if feature_version >= 3 and doping_slog is not None:
        doping_feat = doping_slog.reshape(-1, 1).astype(np.float32)
    else:
        doping_feat = (np.sign(doping) * np.log10(np.abs(doping) + 1.0)).reshape(-1, 1).astype(np.float32)
    xmole = xmole.reshape(-1, 1).astype(np.float32)

    # 5. contact flags from the contact regions' triangle elements
    contact_flags = np.zeros((n, 3), dtype=np.float32)
    for cname, cidx in CONTACT_FLAG_MAP.items():
        el_idx = mesh.region_element_indices.get(cname)
        if el_idx is not None and len(el_idx) and mesh.all_elements:
            verts: set = set()
            for gi in el_idx:
                elem = mesh.all_elements[int(gi)]
                if len(elem) != 3:
                    raise ValueError(
                        f"Contact region '{cname}' has a {len(elem)}-vertex element; "
                        "DF-ISE contacts are triangle sheets (element misparse?)")
                verts.update(int(v) for v in elem)
            contact_flags[sorted(verts), cidx] = 1.0
        elif cname in mesh.region_vertex_indices:
            contact_flags[mesh.region_vertex_indices[cname], cidx] = 1.0

    # 6. bias, 7. permittivity, 8. log DelVorWeight
    vg_feat = np.full((n, 1), vg, dtype=np.float32)
    vd_feat = np.full((n, 1), vd, dtype=np.float32)
    if feature_version >= 3:
        eps_tab = np.array([MATERIAL_PERMITTIVITY_V3[m] for m in MATERIALS])
        perm = np.where(mat_enc.sum(1) > 0, mat_enc @ eps_tab, 1.0).astype(np.float32)
    else:
        perm = np.ones(n, dtype=np.float32)
        for mat, eps_r in MATERIAL_PERMITTIVITY.items():
            perm[vertex_mat == MATERIAL_TO_IDX[mat]] = eps_r
    vor = np.nan_to_num(mf.get("DelVorWeight", np.zeros(n)), nan=0.0)
    vor_log = np.log10(np.maximum(vor, 1e-30)).reshape(-1, 1).astype(np.float32)

    x = np.concatenate([coords_norm, mat_enc, doping_feat, xmole, contact_flags,
                        vg_feat, vd_feat, perm.reshape(-1, 1), vor_log], axis=1)

    src, dst = edge_index[0], edge_index[1]
    displacement = coords[dst] - coords[src]
    distance = np.linalg.norm(displacement, axis=1, keepdims=True)
    if feature_version >= 3:
        half = edge_index.shape[1] // 2
        on_iface = _interface_edges(mesh, edge_index[:, :half].T)
        cross = np.concatenate([on_iface, on_iface]).astype(np.float32).reshape(-1, 1)
    else:
        cross = (vertex_mat[src] != vertex_mat[dst]).astype(np.float32).reshape(-1, 1)
    edge_attr = np.concatenate([distance.astype(np.float32),
                                displacement.astype(np.float32), cross], axis=1)

    y = np.stack([np.nan_to_num(sf.get(k, np.zeros(n)), nan=0.0) for k in
                  ("ElectrostaticPotential", "eQuasiFermiPotential", "hQuasiFermiPotential")],
                 axis=1).astype(np.float32)

    data = pyg_data.Data(
        x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr), y=torch.from_numpy(y),
        pos=torch.from_numpy(coords.astype(np.float32)),
        vor_weight=torch.from_numpy(np.nan_to_num(vor, nan=0.0).astype(np.float32)),
        vertex_material=torch.from_numpy(vertex_mat),
    )
    data.vg, data.vd = float(vg), float(vd)
    data.feature_version = feature_version
    if feature_version >= 3:
        semi_any = (mat_enc[:, MATERIAL_TO_IDX["GaN"]] + mat_enc[:, MATERIAL_TO_IDX["AlGaN"]]) > 0
        y_valid = np.ones((n, 3), dtype=bool)
        y_valid[~semi_any, 1:] = False        # Phi_n / Phi_p are placeholders off-semiconductor
        data.semi_any = torch.from_numpy(semi_any)
        data.y_valid = torch.from_numpy(y_valid)
    data.has_solution = bool(sf)
    if params is not None:
        data.params = dict(params)
        data.nfins = int(params.get("Nfins", 1))
        data.fin_pitch = float(fin_pitch) if fin_pitch is not None else 0.0
    return data
