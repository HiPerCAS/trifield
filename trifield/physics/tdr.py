"""Element connectivity (Sentaurus local vertex order) from a TDR grid file.

Sentaurus TDR files (V-2024.03) are Ogawa (Alembic) containers whose array payloads are
zstd-compressed and prefixed by their uncompressed size (uint64). This module reads the bulk
regions' element streams directly; the vertex order of every element is the local numbering
that Sentaurus uses for its box-method data (Measure / Coefficients), which is NOT the order
that a DF-ISE face list implies.

Layout (reverse engineered, V-2024.03): root/2/1/2 holds the geometry's region groups; a bulk
region group's first child has 15 children: [1] name, [5] material, [9] element stream
(int32: type, v0, v1, ...), [12] region-local -> global vertex map (int64).

zstd: uses the ``zstandard`` module or Python's ``compression.zstd`` if importable, otherwise the
``zstd`` command-line tool (must be on PATH).
"""
from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NVERT = {0: 1, 1: 2, 2: 3, 3: 4, 5: 4, 6: 5, 7: 6, 8: 8}
TET = 5


def _zstd_decompress(buf: bytes) -> bytes:
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(buf, max_output_size=1 << 34)
    except ImportError:
        pass
    try:
        from compression import zstd          # Python >= 3.14
        return zstd.decompress(buf)
    except ImportError:
        pass
    if shutil.which("zstd") is None:
        raise RuntimeError("TDR arrays are zstd-compressed: install `zstandard` or put the `zstd` CLI on PATH")
    return subprocess.run(["zstd", "-d", "-c", "-q"], input=buf, capture_output=True, check=True).stdout


class _Ogawa:
    def __init__(self, path):
        self.b = Path(path).read_bytes()
        if self.b[:5] != b"Ogawa":
            raise ValueError(f"{path}: not an Ogawa (TDR V-2024) container")
        self.root = struct.unpack_from("<Q", self.b, 8)[0]

    def group(self, off):
        if off == 0:
            return []
        n = struct.unpack_from("<Q", self.b, off)[0]
        return list(struct.unpack_from("<%dQ" % n, self.b, off + 8))

    @staticmethod
    def isdata(c):
        return bool(c >> 63)

    def data(self, c):
        off = c & ((1 << 63) - 1)
        if off == 0:
            return b""
        n = struct.unpack_from("<Q", self.b, off)[0]
        return self.b[off + 8: off + 8 + n]

    def array(self, c):
        d = self.data(c)
        n = struct.unpack_from("<Q", d, 0)[0]
        out = d[8:]
        if out[:4] == b"\x28\xb5\x2f\xfd":
            out = _zstd_decompress(out)
        if len(out) != n:
            raise ValueError("TDR array size mismatch")
        return out


@dataclass
class TDRRegion:
    name: str
    material: str
    elements: list            # list of global-vertex arrays (Sentaurus local order)
    types: list


def read_bulk_regions(path) -> list[TDRRegion]:
    """Bulk regions of the first geometry, in file order (= Sentaurus element numbering)."""
    o = _Ogawa(path)
    c = o.root
    for i in (2, 1, 2):
        c = o.group(c)[i]
    out = []
    for rc in o.group(c):
        if o.isdata(rc):
            continue
        g = o.group(rc)
        if not g or o.isdata(g[0]):
            continue
        h = o.group(g[0])
        if len(h) != 15:          # interfaces / contacts have a different layout
            continue
        name = o.data(o.group(h[1])[0]).rstrip(b"\0").decode()
        material = o.data(o.group(h[5])[0]).rstrip(b"\0").decode()
        rv = np.frombuffer(o.array(o.group(h[12])[0]), dtype=np.int64)
        a = np.frombuffer(o.array(o.group(o.group(h[9])[0])[0]), dtype=np.int32)
        els, types, p = [], [], 0
        while p < len(a):
            t = int(a[p]); k = NVERT[t]
            els.append(rv[a[p + 1: p + 1 + k]]); types.append(t); p += 1 + k
        out.append(TDRRegion(name, material, els, types))
    return out


def tetrahedra(path):
    """(tets (nE,4) global vertex ids in Sentaurus local order, materials (nE,), regions (nE,))."""
    tets, mats, regs = [], [], []
    for r in read_bulk_regions(path):
        for e, t in zip(r.elements, r.types):
            if t != TET:
                raise NotImplementedError(f"region {r.name}: element type {t}; only tetrahedral meshes are supported")
            tets.append(e); mats.append(r.material); regs.append(r.name)
    return np.asarray(tets, dtype=np.int64), np.asarray(mats), np.asarray(regs)
