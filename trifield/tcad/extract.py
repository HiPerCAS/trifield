"""Turn finished Sentaurus run directories into training graphs.

    python -m trifield.tcad.extract runs/sim_* --out_dir data/graphs --workers 16

For every run directory (rendered ``sde_dvs.cmd`` + ``sdevice*_des.cmd``,
``node_msh.tdr`` and solution TDRs) the mesh is extracted once and one graph
``<run>__<snapshot>.pt`` is written per solution snapshot found:

* the nine output-curve snapshots of ``sdevice1`` (``snap_vg*_vd00``,
  ``snap_vg10_vd*``, ``node1_des``), and
* the transfer-curve snapshots of ``sdevice2`` (``snap2_vgNNNN_vd12``, Vg in mV,
  ``n`` prefix = negative), which give the surrogate the Vd = 12 V slice used
  to warm-start the threshold-voltage search.

The design parameters are read back from the rendered command files, so no
side table is needed. A ``designs.csv`` with the parameters, the snapshot
count and the simulator's own figures of merit (from the ``.plt`` files) is
written next to the graphs.
"""
from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import re
import time
from pathlib import Path

from trifield.params import FIXED, SFIN, snapshot_biases, snapshot_stem

log = logging.getLogger("trifield.tcad.extract")

_SNAP2 = re.compile(r"snap2_vg(n?)(\d{4})_vd(\d+)_des\.tdr$")


def parse_run_params(run_dir) -> dict:
    """Design parameters of a run, recovered from its rendered command files."""
    run_dir = Path(run_dir)
    txt = (run_dir / "sde_dvs.cmd").read_text()

    def define(k):
        m = re.search(rf"\(define\s+{k}\s+([0-9.eE+-]+)\)", txt)
        return float(m.group(1)) if m else None

    def comment(label):
        m = re.search(rf";;\s*([0-9.eE+-]+)\s+{label}", txt)
        return float(m.group(1)) if m else None

    p = {"Tox": define("Tox"), "Hg": define("Hg"), "Fw": define("Fw"),
         "Nfins": int(define("Nfins") or 1),
         "GaN": comment(r"GaN layer thickness \(also"),
         "AlGaN": comment(r"AlGaN layer thickness"),
         "AlGaN_doping": comment(r"AlGaN source/drain spacer doping")}
    p.update(FIXED)
    lext = define("Lext2")
    if lext is not None:
        p["Lext2"] = lext
    missing = [k for k in ("GaN", "AlGaN", "Fw", "Tox", "Hg") if p.get(k) is None]
    if missing:
        raise ValueError(f"{run_dir.name}: cannot recover {missing} from sde_dvs.cmd")
    return p


def solution_snapshots(run_dir, params) -> list[tuple[str, float, float]]:
    """(stem, Vg, Vd) of every solution TDR present in a run directory."""
    run_dir = Path(run_dir)
    out = [(snapshot_stem(tag), vg, vd) for tag, vg, vd in snapshot_biases(params)
           if (run_dir / f"{snapshot_stem(tag)}.tdr").exists()]
    for f in sorted(run_dir.glob("snap2_vg*_vd*_des.tdr")):
        m = _SNAP2.search(f.name)
        if m:
            vg = (-1 if m.group(1) else 1) * int(m.group(2)) / 1000.0
            out.append((f.stem, vg, float(m.group(3))))
    return out


def tcad_foms(run_dir) -> dict:
    """The simulator's own FoMs: Id,max from node1_des.plt, Vth from node2_des.plt."""
    from trifield.tcad.simulation import PLTFile
    run_dir = Path(run_dir)
    out = {"id_max": None, "vth": None}
    try:
        out["id_max"] = PLTFile(run_dir / "node1_des.plt").max("d_total_current")
    except Exception:                                   # noqa: broad-except
        pass
    try:
        out["vth"] = PLTFile(run_dir / "node2_des.plt").find_vth()
    except Exception:                                   # noqa: broad-except
        pass
    return out


def extract_run(run_dir, out_dir, skip_existing: bool = True) -> dict:
    import torch
    from trifield.data.graph_builder import build_graph
    from trifield.data.tdr_extract import ExtractedSimulation, extract_mesh, extract_solution

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    t0 = time.monotonic()
    row = {"name": run_dir.name, "status": "ok", "n_graphs": 0}
    try:
        params = parse_run_params(run_dir)
        row.update(params)
        row.update(tcad_foms(run_dir))
        snaps = solution_snapshots(run_dir, params)
        todo = [s for s in snaps
                if not (skip_existing and (out_dir / f"{run_dir.name}__{s[0]}.pt").exists())]
        if todo:
            mesh, mesh_fields = extract_mesh(run_dir)
            for stem, vg, vd in todo:
                sol = extract_solution(run_dir / f"{stem}.tdr", mesh)
                sim = ExtractedSimulation(mesh=mesh, mesh_fields=mesh_fields,
                                          solution_fields=sol, sim_dir=run_dir)
                g = build_graph(sim, vg=vg, vd=vd, params=params,
                                fin_pitch=float(params["Fw"]) + SFIN)
                g.sim_name, g.snapshot_name = run_dir.name, stem
                torch.save(g, out_dir / f"{run_dir.name}__{stem}.pt")
        row["n_graphs"] = len(snaps)
    except Exception as e:                              # noqa: broad-except
        row["status"] = f"failed: {e}"
    row["sec"] = round(time.monotonic() - t0, 1)
    return row


def _worker(args):
    return extract_run(*args)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sentaurus run directories -> training graphs")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d) for d in args.run_dirs if (Path(d) / "node_msh.tdr").exists()]
    log.info("%d run directories with a mesh", len(dirs))
    rows = []
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(_worker, [(d, out_dir, not args.overwrite) for d in dirs]):
            rows.append(r)
            log.info("%-40s %s %3d graphs %6.1fs", r["name"], r["status"][:40], r["n_graphs"], r["sec"])
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("name", "status"), k))
    with (out_dir / "designs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["name"]))
    ok = sum(r["status"] == "ok" for r in rows)
    log.info("%d/%d runs ok, %d graphs -> %s", ok, len(rows), sum(r["n_graphs"] for r in rows), out_dir)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
