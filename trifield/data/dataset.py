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
