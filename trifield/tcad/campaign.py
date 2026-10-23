"""Parallel Sentaurus campaign: sample designs, simulate, write graphs.

    # training / validation data (full drift-diffusion solves)
    python -m trifield.tcad.campaign --par parameter.par --out_dir data/train \
        --n_per_fins 40 --max_fins 5 --workers 16 --seed 42

    # mesh-only candidate pool for active learning (sde only, ~30 s each)
    python -m trifield.tcad.campaign --par parameter.par --out_dir runs/candidates \
        --n_per_fins 200 --max_fins 5 --workers 16 --seed 7 --mesh_only

Designs are sampled uniformly over ``trifield.params.RANGES`` per fin count,
subject to the fin-spacing constraint (and, for full solves, the
convergence-yield filter of the paper's campaign). A ``campaign_log.csv``
with the sampled parameters, status and simulator FoMs is written next to the
graphs. Use different seeds and output directories for disjoint train /
validation sets.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from trifield.params import sample_design
from trifield.tcad.simulation import MeshOnlySimulation, Simulation, SimulationParams

log = logging.getLogger("trifield.tcad.campaign")


def run_one(task: tuple) -> dict:
    idx, params, par, work_dir, out_dir, timeout, mesh_only, keep = task
    name = f"{'cand' if mesh_only else 'sim'}_{idx:06d}"
    t0 = time.monotonic()
    row = {"index": idx, "name": name, "status": "ok", "id_max": None, "vth": None, "ron": None,
           **params.to_dict()}
    try:
        if mesh_only:
            MeshOnlySimulation(params, par, work_dir, timeout, keep=keep, name=name).run(out_dir, name)
        else:
            res = Simulation(params, par, work_dir, timeout, keep=keep, name=name).run(out_dir, name)
            row["id_max"], row["vth"], row["ron"] = res["id_max"], res["vth"], res["ron"]
            for tpl, t in res["timings"].items():
                key = tpl.split("_")[0]                  # sde / sdevice1 / sdevice2
                row[f"{key}_wall"], row[f"{key}_cpu"] = round(t["wall"], 2), round(t["cpu"], 2)
    except Exception as e:                            # noqa: broad-except
        row["status"] = f"failed: {e}"
        log.error("%s (N_fins=%d) failed after %.0f s: %s", name, params.Nfins, time.monotonic() - t0, e)
    row["wall_sec"] = round(time.monotonic() - t0, 1)
    run_dir = Path(work_dir) / name
    if keep and run_dir.exists():                    # per-run record, survives crashes
        (run_dir / "campaign_row.json").write_text(json.dumps(row))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sentaurus dataset / candidate-pool campaign")
    ap.add_argument("--par", default=None, help="material parameter file (see tcad/README.md)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--work_dir", default="runs")
    ap.add_argument("--n_per_fins", type=int, default=10)
    ap.add_argument("--min_fins", type=int, default=1)
    ap.add_argument("--max_fins", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=7200, help="per-simulation wall-time cap [s]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mesh_only", action="store_true", help="sde only (candidate pool)")
    ap.add_argument("--fins", type=int, nargs="+", default=None,
                    help="explicit fin counts (overrides --min_fins/--max_fins)")
    ap.add_argument("--keep", action="store_true", help="keep the Sentaurus run directories")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tasks, idx = [], 0
    fins = args.fins or list(range(args.min_fins, args.max_fins + 1))
    for nfins in fins:
        for _ in range(args.n_per_fins):
            p = sample_design(rng, nfins, strict=not args.mesh_only)
            if p is None:
                continue
            tasks.append((idx, SimulationParams.from_dict(p), args.par, args.work_dir,
                          str(out_dir), args.timeout, args.mesh_only, args.keep))
            idx += 1
    log.info("%d designs, N_fins %s, %d workers", len(tasks), fins, args.workers)

    # resume: designs whose kept run directory already holds a record are not re-run
    rows, todo = [], []
    for t in tasks:
        rec = Path(args.work_dir) / f"{'cand' if args.mesh_only else 'sim'}_{t[0]:06d}" / "campaign_row.json"
        if args.keep and rec.exists():
            rows.append(json.loads(rec.read_text()))
        else:
            todo.append(t)
    if rows:
        log.info("resuming: %d designs already done, %d to run", len(rows), len(todo))
    t0 = time.monotonic()
    with mp.Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_one, todo):
            rows.append(r)
            log.info("%s %s N_fins=%s %.0f s", r["name"], r["status"][:60], r["Nfins"], r["wall_sec"])
    rows.sort(key=lambda r: r["index"])
    log_csv = out_dir / "campaign_log.csv"
    with log_csv.open("w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r)) or ["index"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    log.info("done: %d/%d succeeded in %.0f min; log: %s", ok, len(rows), (time.monotonic() - t0) / 60, log_csv)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
