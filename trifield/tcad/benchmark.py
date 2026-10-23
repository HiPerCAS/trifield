"""Benchmark the warm-start twin against the production Sentaurus pipeline.

    # 1. production reference (keeps the run directories, records per-tool times)
    python -m trifield.tcad.campaign --par parameter.par --out_dir runs/bench_graphs \
        --work_dir runs/bench --fins 1 2 3 4 5 6 8 --n_per_fins 3 --keep --workers 12
    # 2. twin on the same meshes
    python -m trifield.tcad.benchmark --runs runs/bench --log runs/bench_graphs/campaign_log.csv \
        --ckpts ckpt/*.pt --par parameter.par --out runs/bench_twin.csv --workers 4

For every design the twin reuses the production mesh (so ``sde`` is common to
both and reported separately) and computes Id,max, Ron and Vth with warm-started
solves; ``--modes warm cold`` also runs the surrogate-free variant. The output
CSV has, per design and mode, the objectives, their relative / absolute errors
against the production values, and the runtimes (wall and CPU) of both.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

log = logging.getLogger("trifield.tcad.benchmark")

_MODELS = None


def _models(ckpts, device):
    global _MODELS
    if _MODELS is None:
        import torch
        torch.set_num_threads(1)
        from trifield.inference import load_ensemble
        _MODELS = load_ensemble(ckpts, device)
    return _MODELS


def run_design(task) -> list[dict]:
    row, run_dir, ckpts, par, modes, device, scratch, twin_kw = task
    from trifield.tcad.simulation import SimulationParams
    from trifield.tcad.twin import Twin
    params = SimulationParams.from_dict(row)
    out = []
    for mode in modes:
        work = Path(scratch) / f"{row['name']}_{mode}"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        shutil.copy(Path(run_dir) / "node_msh.tdr", work / "node_msh.tdr")
        shutil.copy(Path(run_dir) / "parameter.par", work / "parameter.par")
        rec = {"name": row["name"], "Nfins": int(float(row["Nfins"])), "mode": mode}
        try:
            models = _models(ckpts, device) if mode == "warm" else None
            kw = dict(twin_kw)
            per_mode = kw.pop("cold_kw", {})                 # settings for the cold baseline only
            if mode == "cold":
                kw.update(per_mode)
            cap = kw.pop("refine_cap", 0)
            esc = kw.pop("escalate", False)
            overrides = kw.pop("refine_cfg", {})
            if mode == "warm":
                from trifield.physics import RefineConfig
                cfg = RefineConfig()
                if cap:
                    cfg.newton_maxit, cfg.restarts = cap, 1
                for k, v in overrides.items():
                    setattr(cfg, k, v)
                kw.update(refine_config=cfg, refine_escalate=RefineConfig() if esc else None)
            res = Twin(params, work, models, None, mode, device, **kw).run()
            solves = res.pop("solves")
            rec.update(res)
            rec["solves"] = json.dumps([{k: (round(v, 4) if isinstance(v, float) else v)
                                         for k, v in s.items()} for s in solves])
        except Exception as e:                       # noqa: broad-except
            rec["error"] = str(e)[:200]
        for k in ("id_max", "ron", "vth"):
            ref = row.get(k)
            ref = float(ref) if ref not in (None, "", "None") else None
            rec[f"{k}_tcad"] = ref
            val = rec.get(k)
            if ref is not None and val is not None:
                rec[f"{k}_err"] = (abs(val - ref) if k == "vth" else abs(val - ref) / abs(ref))
        prod = [row.get(f"{t}_wall") for t in ("sdevice1", "sdevice2")]
        prod_cpu = [row.get(f"{t}_cpu") for t in ("sdevice1", "sdevice2")]
        if all(v not in (None, "") for v in prod):
            rec["prod_sdevice_wall"] = sum(float(v) for v in prod)
            rec["prod_sdevice_cpu"] = sum(float(v) for v in prod_cpu)
            rec["sde_wall"] = float(row.get("sde_wall") or 0)
            twin_wall = rec.get("wall_without_mesh")
            if twin_wall:
                rec["speedup_wall"] = rec["prod_sdevice_wall"] / twin_wall
                rec["speedup_cpu"] = rec["prod_sdevice_cpu"] / max(rec.get("sdevice_cpu_sum", 0)
                                                                   + rec.get("t_predict", 0)
                                                                   + rec.get("t_prepare", 0)
                                                                   + rec.get("t_guess", 0)
                                                                   + rec.get("t_refine", 0)
                                                                   + rec.get("t_physics_export", 0), 1e-9)
        out.append(rec)
        if not os.environ.get("TRIFIELD_KEEP_BENCH"):
            shutil.rmtree(work, ignore_errors=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="twin vs production benchmark")
    ap.add_argument("--runs", required=True, help="directory holding the kept production run dirs")
    ap.add_argument("--log", default=None,
                    help="campaign_log.csv of the production campaign (default: the per-run "
                         "campaign_row.json records, so a running campaign can be benchmarked)")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--par", default=None)
    ap.add_argument("--modes", nargs="+", default=["warm"], choices=["warm", "cold"])
    ap.add_argument("--out", default="runs/bench_twin.csv")
    ap.add_argument("--workers", type=int, default=4, help="designs in parallel (each uses up to 3 cores)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scratch", default="runs/bench_scratch")
    ap.add_argument("--feature_version", type=int, default=None, help="graph features of the checkpoints")
    ap.add_argument("--refine_device", default="cpu")
    ap.add_argument("--refine_cap", type=int, default=0, help="refinement Newton budget (0 = full)")
    ap.add_argument("--escalate", action="store_true", help="full refinement + direct retry before cold")
    ap.add_argument("--no_refine", action="store_true")
    ap.add_argument("--only", nargs="*", help="design names")
    ap.add_argument("--twin_kw", default="{}", help="extra Twin keyword arguments as JSON (e.g. '{\"max_parallel\": 4}')")
    ap.add_argument("--cold_kw", default="{}", help="Twin keyword arguments for the cold baseline only, as JSON "
                    "(e.g. '{\"vth_mode\": \"legacy\"}' to keep its original bracket)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.log:
        rows = list(csv.DictReader(open(args.log)))
    else:
        rows = [json.loads(p.read_text()) for p in sorted(Path(args.runs).glob("*/campaign_row.json"))]
    rows = [r for r in rows if r["status"] == "ok"]
    tasks = []
    for r in rows:
        rd = Path(args.runs) / r["name"]
        if (rd / "node_msh.tdr").exists():
            if args.only and r["name"] not in args.only:
                continue
            tw = dict(feature_version=args.feature_version, refine=not args.no_refine,
                      refine_device=args.refine_device, refine_cap=args.refine_cap, escalate=args.escalate)
            tw.update(json.loads(args.twin_kw))
            tw["cold_kw"] = json.loads(args.cold_kw)
            tasks.append((r, rd, args.ckpts, args.par, args.modes, args.device, args.scratch, tw))
    log.info("%d designs x %s", len(tasks), args.modes)
    recs = []
    t0 = time.monotonic()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for res in pool.imap_unordered(run_design, tasks):
            for r in res:
                recs.append(r)
                log.info("%-12s %-4s fins=%d  Id err %s  Vth err %s  speedup %s  %s", r["name"], r["mode"],
                         r["Nfins"], _fmt(r.get("id_max_err"), "%"), _fmt(r.get("vth_err"), "V"),
                         _fmt(r.get("speedup_wall"), "x"), r.get("error", ""))
    keys = list(dict.fromkeys(k for r in recs for k in r))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(recs)
    log.info("done in %.0f min -> %s", (time.monotonic() - t0) / 60, args.out)


def _fmt(v, unit):
    if v is None:
        return "-"
    return f"{100 * v:.2f}%" if unit == "%" else (f"{v * 1e3:.0f}mV" if unit == "V" else f"{v:.1f}x")


if __name__ == "__main__":
    main()
