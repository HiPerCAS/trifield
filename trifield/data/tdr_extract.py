"""Sentaurus TDR -> DF-ISE text -> numpy.

``tdx`` (shipped with Sentaurus) converts a .tdr snapshot to the DF-ISE text
format (.grd mesh + .dat fields); the parsers below turn those into numpy
arrays. The tool is located from ``$SENTAURUS_BIN``, ``$STROOT/bin`` or PATH.

Library use:
    from trifield.data.tdr_extract import extract_tdr
    sim = extract_tdr(sim_dir)          # mesh + mesh fields + solution fields

Command line:
    python -m trifield.data.tdr_extract <sim_dir>
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def find_tdx() -> Optional[Path]:
    """Locate the Sentaurus ``tdx`` binary (None if Sentaurus is not available)."""
    for env in ("SENTAURUS_BIN",):
        p = os.environ.get(env)
        if p and (Path(p) / "tdx").exists():
            return Path(p) / "tdx"
    stroot = os.environ.get("STROOT")
    if stroot and (Path(stroot) / "bin" / "tdx").exists():
        return Path(stroot) / "bin" / "tdx"
    which = shutil.which("tdx")
    return Path(which) if which else None


# =========================================================================
# 1. TDR → DF-ISE conversion
# =========================================================================

def _run_tdx(args: List[str], cwd: Path) -> None:
    """Run a tdx command, raising on failure."""
    tdx = find_tdx()
    if tdx is None:
        raise FileNotFoundError(
            "Sentaurus 'tdx' not found: set SENTAURUS_BIN (or STROOT) or add it to PATH")
    cmd = [str(tdx)] + args
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"tdx failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr.strip()}"
        )


def convert_tdr_to_dfise(
    tdr_path: Path,
    output_dir: Path,
    output_base: str = "converted",
) -> Tuple[Path, Path]:
    """Convert a .tdr file to DF-ISE .grd + .dat text files.

    Returns (grd_path, dat_path).
    """
    tdr_path = Path(tdr_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: coordinate system → traditional
    trad_tdr = output_dir / f"{output_base}_trad.tdr"
    _run_tdx(
        ["--tdr-change-cs", "--traditional", str(tdr_path), str(trad_tdr)],
        cwd=output_dir,
    )

    # Step 2: tdr → DF-ISE  (tdx writes output relative to its CWD)
    _run_tdx(
        ["--tdr2dfise", str(trad_tdr), output_base],
        cwd=output_dir,
    )

    grd = output_dir / f"{output_base}.grd"
    dat = output_dir / f"{output_base}.dat"

    # Clean up intermediate trad TDR
    trad_tdr.unlink(missing_ok=True)

    if not grd.exists() or not dat.exists():
        raise FileNotFoundError(
            f"Expected DF-ISE files not found in {output_dir}.\n"
            f"  .grd exists: {grd.exists()}\n"
            f"  .dat exists: {dat.exists()}"
        )
    return grd, dat


# =========================================================================
# 2. DF-ISE .grd / .dat shared helpers
# =========================================================================

def _find_matching_brace(text: str, open_pos: int) -> int:
    """Return index of the closing '}' matching the '{' at *open_pos*."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1


# =========================================================================
# 3. DF-ISE .grd parser
# =========================================================================

@dataclass
class DFISEMesh:
    """Parsed contents of a DF-ISE .grd file."""
    vertices: np.ndarray              # (N_vertices, 3) float64
    elements: np.ndarray              # (N_tets, 4) int32  (0-based vertex indices)
    element_region_ids: np.ndarray    # (N_tets,) int32  region index per tet
    region_names: List[str]           # ordered region name list
    region_materials: List[str]       # material per region (same order)
    region_vertex_indices: Dict[str, np.ndarray] = field(default_factory=dict)
    # ^ sorted vertex indices per region — needed for .dat stitching
    n_vertices: int = 0
    n_edges: int = 0
    n_faces: int = 0
    n_elements: int = 0
    # Optional extras for visualization exports (kept out of the training
    # pipeline): every element as its unique vertex list (2D contact/interface
    # elements included), and each region's global element indices.
    all_elements: List[List[int]] = field(default_factory=list)
    region_element_indices: Dict[str, np.ndarray] = field(default_factory=dict)
    # region-wise vertex datasets, {field: {region: values aligned with
    # region_vertex_indices[region]}}; DF-ISE has one vertex per location shared by
    # all regions touching it, and material-dependent fields differ per region there
    region_fields: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)


def _resolve_signed_index(idx: int) -> Tuple[int, bool]:
    """Resolve a signed 1-style index: positive → (idx, False), negative → (abs-1, True)."""
    if idx < 0:
        return (-idx - 1, True)
    return (idx, False)


def parse_grd(path: Path) -> DFISEMesh:
    """Parse a DF-ISE .grd text file.

    Follows the edge → face → element resolution chain as used by
    Sentaurus DF-ISE (see allpix-squared DFISEParser for reference).
    """
    text = Path(path).read_text()

    # ---- Info header ----
    info_match = re.search(r"Info\s*\{(.*?)\}", text, re.DOTALL)
    if not info_match:
        raise ValueError("No Info section in .grd file")
    info = info_match.group(1)

    n_vertices = int(re.search(r"nb_vertices\s*=\s*(\d+)", info).group(1))
    n_edges = int(re.search(r"nb_edges\s*=\s*(\d+)", info).group(1))
    n_faces = int(re.search(r"nb_faces\s*=\s*(\d+)", info).group(1))
    n_elements = int(re.search(r"nb_elements\s*=\s*(\d+)", info).group(1))

    dim_match = re.search(r"dimension\s*=\s*(\d+)", info)
    dimension = int(dim_match.group(1)) if dim_match else 3

    # Parse region names from Info
    regions_match = re.search(
        r'regions\s*=\s*\[\s*(.*?)\s*\]', info, re.DOTALL
    )
    region_names = re.findall(r'"([^"]+)"', regions_match.group(1))

    # Parse materials list from Info
    mat_match = re.search(
        r'materials\s*=\s*\[\s*(.*?)\s*\]', info, re.DOTALL
    )
    region_materials = mat_match.group(1).split()

    # ---- Vertices ----
    vert_match = re.search(
        r"Vertices\s*\(\d+\)\s*\{(.*?)\}", text, re.DOTALL
    )
    vert_vals = np.fromstring(
        vert_match.group(1), sep=' ', dtype=np.float64
    ).reshape(n_vertices, dimension)
    if dimension == 2:
        # Pad 2D coords to 3D with x=0
        vert_vals = np.column_stack([
            np.zeros(n_vertices), vert_vals
        ])

    # ---- Edges ----
    edges_match = re.search(
        r"Edges\s*\(\d+\)\s*\{(.*?)\}", text, re.DOTALL
    )
    edge_vals = np.fromstring(
        edges_match.group(1), sep=' ', dtype=np.int64
    ).reshape(-1, 2)
    # edges[i] = (v1, v2)

    # ---- Faces ----
    # Each face line: <n_edges> <signed_edge_idx> ...
    # Resolve to unique ordered vertex indices per face.
    faces_match = re.search(
        r"Faces\s*\(\d+\)\s*\{(.*?)\}", text, re.DOTALL
    )
    faces_text = faces_match.group(1).strip()
    faces: List[List[int]] = []  # faces[i] = list of vertex indices

    for line in faces_text.splitlines():
        nums = line.split()
        if not nums:
            continue
        pos = 0
        while pos < len(nums):
            n_edges_in_face = int(nums[pos])
            pos += 1
            face_verts: List[int] = []
            for _ in range(n_edges_in_face):
                eidx, swap = _resolve_signed_index(int(nums[pos]))
                pos += 1
                e = edge_vals[eidx]
                v1, v2 = (int(e[1]), int(e[0])) if swap else (int(e[0]), int(e[1]))
                # Connect with previous: if last vertex == v2, swap
                if face_verts and face_verts[-1] == v2:
                    v1, v2 = v2, v1
                face_verts.append(v1)
                face_verts.append(v2)
            # Deduplicate consecutive + close the loop
            seen = []
            prev = None
            for v in face_verts:
                if v != prev:
                    seen.append(v)
                    prev = v
            if seen and seen[0] == seen[-1]:
                seen.pop()
            faces.append(seen)

    # ---- Elements ----
    # Each line: <type> <signed_face/edge_indices...>
    # Type 5 (tet) has 4 face indices; resolve to unique vertex indices.
    elem_match = re.search(
        r"Elements\s*\(\d+\)\s*\{(.*?)\}", text, re.DOTALL
    )
    elem_text = elem_match.group(1).strip()

    # Store all elements as vertex index lists; track which are tets
    all_elements: List[List[int]] = []  # global element list
    tet_global_indices: List[int] = []  # which global indices are tets
    tets: List[List[int]] = []

    _ELEM_NREFS = {0: 1, 1: 2, 2: 3, 3: 4, 5: 4, 6: 5, 7: 5, 8: 6}
    # DF-ISE element references depend on the element type:
    #   0 (point) / 1 (segment): vertex indices
    #   2 (triangle) / 3 (quad): EDGE indices
    #   5 (tet) / 6 (pyramid) / 7 (prism) / 8 (brick): FACE indices

    for line in elem_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        pos = 0
        while pos < len(parts):
            etype = int(parts[pos])
            pos += 1
            n_refs = _ELEM_NREFS.get(etype, 0)
            verts: List[int] = []
            for _ in range(n_refs):
                fidx, reverse = _resolve_signed_index(int(parts[pos]))
                pos += 1
                if etype <= 1:
                    # References are vertex indices (sign has no meaning)
                    verts.append(fidx)
                elif etype in (2, 3):
                    # References are edge indices
                    e = edge_vals[fidx]
                    v1, v2 = int(e[0]), int(e[1])
                    if reverse:
                        v1, v2 = v2, v1
                    verts.extend([v1, v2])
                else:
                    # References are face indices
                    face = list(faces[fidx])
                    if reverse:
                        face = [face[0]] + face[1:][::-1]
                    verts.extend(face)
            # Deduplicate to get unique vertex set
            unique_verts = list(dict.fromkeys(verts))
            global_idx = len(all_elements)
            all_elements.append(unique_verts)
            if etype == 5:
                tet_global_indices.append(global_idx)
                tets.append(unique_verts)

    elements = np.array(tets, dtype=np.int32) if tets else np.empty((0, 4), dtype=np.int32)

    # ---- Region → element mapping ----
    element_region_ids = np.full(len(tets), -1, dtype=np.int32)
    # Map global element index → tet local index
    global_to_tet = {g: i for i, g in enumerate(tet_global_indices)}

    name_to_idx = {name: i for i, name in enumerate(region_names)}

    # Per-region sorted unique vertex indices (needed for .dat stitching)
    region_vertex_indices: Dict[str, np.ndarray] = {}
    # Per-region global element indices (needed for exact contact faces)
    region_element_indices: Dict[str, np.ndarray] = {}

    # Find all Region blocks
    region_header = re.compile(r'Region\s*\("([^"]+)"\)\s*\{')
    for rmatch in region_header.finditer(text, elem_match.end()):
        rname = rmatch.group(1)
        open_pos = rmatch.end() - 1
        close_pos = _find_matching_brace(text, open_pos)
        if close_pos < 0:
            continue
        rbody = text[open_pos + 1 : close_pos]

        el_match = re.search(
            r'Elements\s*\(\d+\)\s*\{(.*?)\}', rbody, re.DOTALL
        )
        if el_match is None:
            continue

        el_global_indices = [int(x) for x in el_match.group(1).split()]
        rid = name_to_idx.get(rname, -1)

        region_verts: set = set()
        for gi in el_global_indices:
            if gi < len(all_elements):
                region_verts.update(all_elements[gi])
            # Map tets to their region
            if gi in global_to_tet:
                element_region_ids[global_to_tet[gi]] = rid

        region_vertex_indices[rname] = np.array(
            sorted(region_verts), dtype=np.int64
        )
        region_element_indices[rname] = np.array(
            el_global_indices, dtype=np.int64
        )

    return DFISEMesh(
        vertices=vert_vals,
        elements=elements,
        element_region_ids=element_region_ids,
        region_names=region_names,
        region_materials=region_materials,
        region_vertex_indices=region_vertex_indices,
        n_vertices=n_vertices,
        n_edges=n_edges,
        n_faces=n_faces,
        n_elements=n_elements,
        all_elements=all_elements,
        region_element_indices=region_element_indices,
    )


# =========================================================================
# 3. DF-ISE .dat parser
# =========================================================================

@dataclass
class DFISEDataset:
    """A single named dataset from a .dat file."""
    name: str
    function: str
    dtype: str          # "scalar" or "vector"
    location: str       # "vertex", "edge", or "element"
    region: str         # validity region name
    values: np.ndarray  # (n,) for scalar, (n, 3) for vector


def parse_dat(path: Path) -> List[DFISEDataset]:
    """Parse a DF-ISE .dat text file, returning all datasets."""
    text = Path(path).read_text()

    datasets: List[DFISEDataset] = []

    # Find each Dataset header; then use brace counting to get the full body
    # (simple regex fails because nested Values { } confuses non-greedy match)
    ds_header = re.compile(r'Dataset\s*\("([^"]+)"\)\s*\{')

    for m in ds_header.finditer(text):
        name = m.group(1)
        open_pos = m.end() - 1  # position of the '{'
        close_pos = _find_matching_brace(text, open_pos)
        if close_pos < 0:
            continue
        body = text[open_pos + 1 : close_pos]

        func = re.search(r'function\s*=\s*(\S+)', body).group(1)
        dtype = re.search(r'type\s*=\s*(\S+)', body).group(1)
        dim = int(re.search(r'dimension\s*=\s*(\d+)', body).group(1))
        location = re.search(r'location\s*=\s*(\S+)', body).group(1)

        validity_match = re.search(
            r'validity\s*=\s*\[\s*(.*?)\s*\]', body, re.DOTALL
        )
        region = ""
        if validity_match:
            regions_in_validity = re.findall(
                r'"([^"]+)"', validity_match.group(1)
            )
            region = regions_in_validity[0] if regions_in_validity else ""

        # Parse Values block
        val_match = re.search(
            r'Values\s*\((\d+)\)\s*\{(.*?)\}', body, re.DOTALL
        )
        if val_match is None:
            continue

        raw = val_match.group(2)
        vals = np.fromstring(raw, sep=' ', dtype=np.float64)

        if dtype == "vector":
            vals = vals.reshape(-1, dim)

        datasets.append(DFISEDataset(
            name=name, function=func, dtype=dtype,
            location=location, region=region, values=vals,
        ))

    return datasets


def collect_node_fields(
    datasets: List[DFISEDataset],
    mesh: DFISEMesh,
) -> Dict[str, np.ndarray]:
    """Assemble per-node (vertex) fields from per-region datasets.

    Many fields appear once per region in the .dat file with values only
    for nodes belonging to that region. This function stitches them into
    full-mesh arrays of shape (n_vertices,) or (n_vertices, dim).

    The .dat values for each region are ordered by the sorted unique vertex
    indices of that region (as computed in parse_grd and stored in
    mesh.region_vertex_indices).

    Returns a dict mapping field name to a numpy array.
    """
    # Group datasets by (field name, type) — some functions appear as both
    # scalar and vector (e.g. eCurrentDensity scalar for contacts, vector for bulk)
    field_datasets: Dict[Tuple[str, str], List[DFISEDataset]] = {}
    for ds in datasets:
        if ds.location != "vertex":
            continue
        key = (ds.function, ds.dtype)
        if key not in field_datasets:
            field_datasets[key] = []
        field_datasets[key].append(ds)

    # Assemble full-mesh arrays
    result: Dict[str, np.ndarray] = {}
    n = mesh.n_vertices

    for (fname, ftype), ds_list in field_datasets.items():
        # Suffix vector fields to avoid name collisions with scalars
        out_name = fname if ftype == "scalar" else f"{fname}_vec"
        if fname not in result and ftype == "scalar":
            out_name = fname
        elif fname not in result and ftype == "vector":
            out_name = fname

        is_vector = ftype == "vector"
        vdim = ds_list[0].values.shape[1] if is_vector else 0
        shape = (n, vdim) if is_vector else (n,)
        arr = np.full(shape, np.nan, dtype=np.float64)

        for ds in ds_list:
            if ds.region not in mesh.region_vertex_indices:
                continue
            vidx = mesh.region_vertex_indices[ds.region]
            if len(vidx) != len(ds.values):
                continue
            arr[vidx] = ds.values

        result[out_name] = arr

    return result


REGION_FIELDS = ("DopingConcentration", "xMoleFraction")


def collect_region_fields(datasets: List[DFISEDataset], mesh: DFISEMesh,
                          names=REGION_FIELDS) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-region scalar vertex datasets (not stitched), for volume-weighted features."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for ds in datasets:
        if ds.name in names and ds.location == "vertex" and ds.dtype == "scalar" \
                and ds.region in mesh.region_vertex_indices \
                and len(mesh.region_vertex_indices[ds.region]) == len(ds.values):
            out.setdefault(ds.name, {})[ds.region] = np.nan_to_num(ds.values, nan=0.0)
    return out


# =========================================================================
# 4. High-level extraction API
# =========================================================================

@dataclass
class ExtractedSimulation:
    """All data extracted from a single simulation directory."""
    mesh: DFISEMesh
    mesh_fields: Dict[str, np.ndarray]    # from node_msh.tdr
    solution_fields: Dict[str, np.ndarray]  # from node1_des.tdr
    sim_dir: Path = field(default_factory=Path)


def extract_mesh(
    sim_dir: Path,
    mesh_tdr: str = "node_msh.tdr",
    cache_dir: Optional[Path] = None,
) -> Tuple[DFISEMesh, Dict[str, np.ndarray]]:
    """Extract only mesh topology and mesh-level fields.

    Returns (mesh, mesh_fields).  Useful when processing multiple solution
    TDRs that share the same mesh.
    """
    sim_dir = Path(sim_dir).resolve()
    cleanup = cache_dir is None
    work_dir = (
        Path(cache_dir).resolve() if cache_dir
        else Path(tempfile.mkdtemp(prefix="trifield_dfise_"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        mesh_grd, mesh_dat = convert_tdr_to_dfise(
            sim_dir / mesh_tdr, work_dir, "mesh"
        )
        mesh = parse_grd(mesh_grd)
        mesh_datasets = parse_dat(mesh_dat)
        mesh_fields = collect_node_fields(mesh_datasets, mesh)
        mesh.region_fields = collect_region_fields(mesh_datasets, mesh)
        return mesh, mesh_fields
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


def extract_solution(
    tdr_path: Path,
    mesh: DFISEMesh,
    cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Extract solution fields from a single solution TDR.

    Parameters
    ----------
    tdr_path : absolute path to the solution .tdr file
    mesh : already-parsed DFISEMesh (shared across snapshots)
    cache_dir : if given, store DF-ISE intermediates here
    """
    tdr_path = Path(tdr_path).resolve()
    cleanup = cache_dir is None
    work_dir = (
        Path(cache_dir).resolve() if cache_dir
        else Path(tempfile.mkdtemp(prefix="trifield_dfise_"))
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        _, sol_dat = convert_tdr_to_dfise(
            tdr_path, work_dir, tdr_path.stem
        )
        sol_datasets = parse_dat(sol_dat)
        return collect_node_fields(sol_datasets, mesh)
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


def extract_tdr(
    sim_dir: Path,
    mesh_tdr: str = "node_msh.tdr",
    solution_tdr: str = "node1_des.tdr",
    cache_dir: Optional[Path] = None,
) -> ExtractedSimulation:
    """Extract mesh and solution data from a simulation directory.

    Parameters
    ----------
    sim_dir : Path to a sim_<ts>_<rand>/ directory containing .tdr files
    mesh_tdr : filename of the mesh TDR within sim_dir
    solution_tdr : filename of the solution TDR within sim_dir
    cache_dir : if given, store DF-ISE intermediates here; otherwise use a
                temporary directory that is cleaned up after parsing.
    """
    sim_dir = Path(sim_dir).resolve()
    mesh, mesh_fields = extract_mesh(sim_dir, mesh_tdr, cache_dir)
    solution_fields = extract_solution(
        sim_dir / solution_tdr, mesh, cache_dir
    )
    return ExtractedSimulation(
        mesh=mesh,
        mesh_fields=mesh_fields,
        solution_fields=solution_fields,
        sim_dir=sim_dir,
    )


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m trifield.data.tdr_extract <sim_dir>")
        sys.exit(1)

    sim = extract_tdr(Path(sys.argv[1]))

    print(f"Mesh: {sim.mesh.n_vertices} vertices, "
          f"{len(sim.mesh.elements)} tets")
    print(f"Mesh fields: {list(sim.mesh_fields.keys())}")
    print(f"Solution fields: {list(sim.solution_fields.keys())}")

    # Quick sanity check
    for name, arr in sim.solution_fields.items():
        valid = np.count_nonzero(~np.isnan(arr) if arr.ndim == 1
                                 else ~np.isnan(arr[:, 0]))
        print(f"  {name}: shape={arr.shape}, valid={valid}/{sim.mesh.n_vertices}")
