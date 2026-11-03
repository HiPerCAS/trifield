"""Dataset over pre-built mesh graphs (one ``torch_geometric.data.Data`` per .pt).

Each file holds one (design, bias) snapshot. File names follow
``<design>__<snapshot>.pt`` so the nine bias snapshots of a design can be
grouped with ``group_by_design``.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch


def load_graph(path):
    """Load one PyG ``Data`` graph (pickled, hence ``weights_only=False``)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("sim_time_sec",):            # drop optional metadata that breaks collation
        if k in data:
            del data[k]
    return data


def design_name(path) -> str:
    """``<design>__<snapshot>.pt`` -> ``<design>``."""
    stem = Path(path).stem
    return stem.split("__")[0] if "__" in stem else stem


def group_by_design(paths) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(Path(x) for x in paths):
        groups[design_name(p)].append(p)
    return dict(groups)


class DeviceMeshDataset(torch.utils.data.Dataset):
    """Lazy dataset over ``*.pt`` graphs under ``root`` (``pattern`` may be recursive)."""

    def __init__(self, root, pattern: str = "*.pt", cache: bool = False):
        self.root = Path(root).resolve()
        self.files: list[Path] = sorted(self.root.glob(pattern))
        # cache=True keeps every graph in RAM after the first load (epochs then
        # skip disk and unpickling entirely; ~1 GB per 250 graphs of 1-8 fins)
        self._cache: dict[int, object] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        if self._cache is None:
            return load_graph(self.files[idx])
        if idx not in self._cache:
            self._cache[idx] = load_graph(self.files[idx])
        return self._cache[idx]

    def designs(self) -> dict[str, list[Path]]:
        return group_by_design(self.files)

    def __repr__(self) -> str:
        return f"DeviceMeshDataset(root={self.root}, n={len(self.files)})"


import re as _re

_SNAP = _re.compile(r"snap_vg(\d\d)_vd(\d\d)_des$")
_SNAP2 = _re.compile(r"snap2_vg(n?)(\d{4})_vd(\d+)_des$")


def snapshot_bias(stem: str, vg11: float = 5.0) -> tuple[float, float] | None:
    """(Vg, Vd) encoded in a snapshot file stem (see trifield.tcad.extract)."""
    if stem == "node1_des":
        return vg11, 12.0
    m = _SNAP.match(stem)
    if m:
        return int(m.group(1)) / 10 * vg11, float(int(m.group(2)))
    m = _SNAP2.match(stem)
    if m:
        return (-1 if m.group(1) else 1) * int(m.group(2)) / 1000, float(m.group(3))
    return None


class BaselineDeltaDataset(DeviceMeshDataset):
    """Delta learning from an exact zero-drain-bias state.

    Every graph at (Vg, Vd > 0) is paired with the same design's solution at
    (Vg, 0). That baseline (psi, Phi_n, Phi_p) is appended to the node features
    (in_dim 17 -> 20) and the target becomes the change ``y - y_base``. At
    inference the baseline comes from a warm-started Sentaurus solve at Vd = 0,
    which the surrogate predicts accurately (the quasi-Fermi levels are exactly
    zero there), so the network only has to learn the drain-bias response, not
    the device electrostatics. Graphs without a matching baseline are skipped.
    """

    def __init__(self, root, pattern: str = "*.pt", cache: bool = False, vg11: float = 5.0):
        super().__init__(root, pattern, cache)
        base = {}
        for f in self.files:
            design, stem = f.stem.split("__", 1)
            b = snapshot_bias(stem, vg11)
            if b and abs(b[1]) < 1e-9:
                base[(design, round(b[0], 3))] = f
        pairs = []
        for f in self.files:
            design, stem = f.stem.split("__", 1)
            b = snapshot_bias(stem, vg11)
            if b and b[1] > 0 and (design, round(b[0], 3)) in base:
                pairs.append((f, base[(design, round(b[0], 3))]))
        self.pairs = pairs
        self.files = [p[0] for p in pairs]
        self._bcache = {} if cache else None

    def _base(self, path):
        if self._bcache is None:
            return load_graph(path)
        if path not in self._bcache:
            self._bcache[path] = load_graph(path)
        return self._bcache[path]

    def __getitem__(self, idx: int):
        data = super().__getitem__(idx)
        if getattr(data, "y_base", None) is not None:          # cached, already transformed
            return data
        base = self._base(self.pairs[idx][1])
        data = data.clone()
        data.x = torch.cat([data.x, base.y], dim=1)
        data.y_base = base.y
        data.y = data.y - base.y
        if self._cache is not None:
            self._cache[idx] = data
        return data



class HybridDataset(DeviceMeshDataset):
    """Ablation helper: version-3 graphs with selected feature groups taken from the
    matching older-version graph (same file name, identical node / edge order)."""

    def __init__(self, root, other_dirs, x_cols=(), edge_cols=(), drop_valid=False,
                 pattern: str = "*.pt", cache: bool = False):
        super().__init__(root, pattern, cache)
        self.other = {}
        for d in other_dirs:
            for f in Path(d).glob("*.pt"):
                self.other.setdefault(f.name, f)
        self.x_cols, self.edge_cols, self.drop_valid = list(x_cols), list(edge_cols), drop_valid

    def __getitem__(self, idx: int):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        data = load_graph(self.files[idx])
        if self.x_cols or self.edge_cols:
            old = load_graph(self.other[self.files[idx].name])
            if self.x_cols:
                data.x[:, self.x_cols] = old.x[:, self.x_cols]
            if self.edge_cols:
                data.edge_attr[:, self.edge_cols] = old.edge_attr[:, self.edge_cols]
        if self.drop_valid:
            for k in ("y_valid", "semi_any"):
                if k in data:
                    del data[k]
        if self._cache is not None:
            self._cache[idx] = data
        return data


# bias points of a twin-labelled design (Vd = 0 gate sweep, on-state drain sweep, off-state / near threshold)
STANDARD_BIASES = ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0), (5.0, 3.0), (5.0, 6.0), (5.0, 9.0),
                   (5.0, 12.0), (0.0, 12.0), (0.5, 12.0))


class UnlabeledBiasDataset(torch.utils.data.Dataset):
    """Mesh-only items for physics-only training: one base graph per design (``<design>__mesh.pt``, built like the
    twin's prediction graph) at ``biases``; the bias columns (13, 14) and ``vg`` / ``vd`` are set per item, targets
    are zero and ``y_valid`` is False everywhere (the data loss ignores these nodes). ``per_design`` > 0 draws that
    many biases per design (seeded) instead of all of them."""

    def __init__(self, root, biases=STANDARD_BIASES, per_design: int = 0, seed: int = 0, exclude=()):
        import random
        self.root = Path(root).resolve()
        self.base_files = [f for f in sorted(self.root.glob("*__mesh.pt")) if design_name(f) not in set(exclude)]
        rng = random.Random(seed)
        self.items = []
        for f in self.base_files:
            bs = list(biases) if per_design <= 0 else rng.sample(list(biases), min(per_design, len(biases)))
            self.items += [(f, float(vg), float(vd)) for vg, vd in bs]
        self._cache: dict = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        f, vg, vd = self.items[idx]
        if f not in self._cache:
            self._cache[f] = load_graph(f)
        g = self._cache[f].clone()
        g.x = g.x.clone()
        g.x[:, 13] = vg
        g.x[:, 14] = vd
        g.vg, g.vd = vg, vd
        g.y = torch.zeros_like(g.y)
        g.y_valid = torch.zeros(g.y.shape, dtype=torch.bool)
        g.snapshot_name = f"unl_vg{vg:+.3f}_vd{vd:.3f}"
        return g

    def __repr__(self):
        return f"UnlabeledBiasDataset(root={self.root}, designs={len(self.base_files)}, n={len(self.items)})"


class MixedDataset(torch.utils.data.Dataset):
    """Labelled graphs (y_valid = True where absent) followed by unlabelled items."""

    def __init__(self, labeled, unlabeled):
        self.labeled, self.unlabeled = labeled, unlabeled

    def __len__(self):
        return len(self.labeled) + len(self.unlabeled)

    def __getitem__(self, idx):
        if idx < len(self.labeled):
            g = self.labeled[idx]
            if getattr(g, "y_valid", None) is None:
                g = g.clone()
                g.y_valid = torch.ones(g.y.shape, dtype=torch.bool)
            return g
        return self.unlabeled[idx - len(self.labeled)]

    def __repr__(self):
        return f"MixedDataset({self.labeled!r} + {self.unlabeled!r})"


class PseudoLabelDataset(torch.utils.data.Dataset):
    """Mesh-only designs labelled by the exact refinement (converged ``trifield.physics`` solutions, see
    runs/agents/phys/scripts/pseudo_labels.py): base graph ``<design>__mesh.pt`` + targets ``<pseudo_dir>/<file>.npy``
    listed in ``<pseudo_dir>/index.jsonl`` (rows with conv = true)."""

    def __init__(self, pseudo_dir, mesh_dir):
        import json
        self.pdir, self.mdir = Path(pseudo_dir), Path(mesh_dir)
        rows = [json.loads(l) for l in open(self.pdir / "index.jsonl")]
        self.items = [(r["design"], float(r["vg"]), float(r["vd"]), r["file"]) for r in rows if r.get("conv") and r.get("file")]
        self._cache: dict = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import numpy as np
        d, vg, vd, f = self.items[idx]
        if d not in self._cache:
            self._cache[d] = load_graph(self.mdir / f"{d}__mesh.pt")
        g = self._cache[d].clone()
        g.x = g.x.clone()
        g.x[:, 13] = vg
        g.x[:, 14] = vd
        g.vg, g.vd = vg, vd
        g.y = torch.from_numpy(np.load(self.pdir / f))
        g.y_valid = torch.ones(g.y.shape, dtype=torch.bool)
        g.snapshot_name = f"pseudo_vg{vg:+.3f}_vd{vd:.3f}"
        return g

    def __repr__(self):
        return f"PseudoLabelDataset({self.pdir}, n={len(self.items)})"


class ConcatGraphs(torch.utils.data.Dataset):
    """Several graph datasets back to back; adds y_valid = True where a graph has none."""

    def __init__(self, *parts):
        self.parts = parts

    def __len__(self):
        return sum(len(p) for p in self.parts)

    def __getitem__(self, idx):
        for p in self.parts:
            if idx < len(p):
                g = p[idx]
                if getattr(g, "y_valid", None) is None:
                    g = g.clone()
                    g.y_valid = torch.ones(g.y.shape, dtype=torch.bool)
                return g
            idx -= len(p)
        raise IndexError(idx)

    def __repr__(self):
        return "ConcatGraphs(" + " + ".join(repr(p) for p in self.parts) + ")"
