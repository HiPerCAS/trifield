"""Physics pseudo-labels: training targets for mesh-only designs from the exact refinement instead of Sentaurus.

The surrogate's guess is refined on Sentaurus' discrete equations (``trifield.physics.Solver``, full RefineConfig: Newton
to merit 1e-9); converged refinements are the discrete solution (1.6-4 mV psi RMS from Sentaurus at every Vg >= 0 test
point from the raw guess) at ~10 s CPU per bias point and no Sentaurus run. In paired fine-tunes (RESULTS.md section 9)
these labels for 249 unlabelled designs cut the psi error of 29 held-out designs of the same pool by 13% (28/29 designs),
while an exact-operator physics LOSS on the same designs did not help.

    # 1. mesh-only base graphs of designs with per-mesh exports (<pool>/<design>/twin/{msh.grd,msh.dat,physics/})
    python -m trifield.pseudolabel graphs --pool runs/pool5 --out data/pool5_mesh
    # 2. refined targets (4 random standard biases per design), then train with --unlabeled_dir / --pseudo_dir
    python -m trifield.pseudolabel labels --ckpts ckpt/*.pt --mesh_dir data/pool5_mesh --exports runs/pool5 \\
        --out data/pool5_pseudo --per_design 4 --workers 12
    python -m trifield.train ... --unlabeled_dir data/pool5_mesh --pseudo_dir data/pool5_pseudo
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch


def build_mesh_graphs(pool: Path, out: Path, names=None) -> int:
    """<design>__mesh.pt for every design under ``pool`` with a twin mesh and a per-mesh export (built like
    Twin.predict's graph; node order checked against the export)."""
    from trifield.data.graph_builder import build_graph
    from trifield.data.tdr_extract import ExtractedSimulation, collect_node_fields, parse_dat, parse_grd
    from trifield.params import SFIN
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(Path(pool).glob("*")):
        w = d / "twin"
        if (names and d.name not in names) or (out / f"{d.name}__mesh.pt").exists():
            continue
        if not ((w / "physics/mesh_data.pkl").exists() and (w / "msh.grd").exists() and (d / "campaign_row.json").exists()):
            continue
        row = json.loads((d / "campaign_row.json").read_text())
        p = {k: row[k] for k in ("GaN", "AlGaN", "Fw", "Tox", "Hg", "AlGaN_doping", "ch_doping", "Vg11", "Lext2", "Nfins")}
        mesh = parse_grd(w / "msh.grd")
        sim = ExtractedSimulation(mesh=mesh, mesh_fields=collect_node_fields(parse_dat(w / "msh.dat"), mesh),
                                  solution_fields={}, sim_dir=d)
        g = build_graph(sim, vg=0.0, vd=0.0, params=p, fin_pitch=float(p["Fw"]) + SFIN)
        g.sim_name, g.snapshot_name = d.name, "mesh"
        md = pickle.loads((w / "physics/mesh_data.pkl").read_bytes())
        if md.n != g.num_nodes or float(np.abs(g.pos.numpy() - md.X).max()) > 1e-6:
            continue
        torch.save(g, out / f"{d.name}__mesh.pt")
        n += 1
    return n


def _init(threads):
    import ctypes, signal
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)          # die with the parent
    torch.set_num_threads(threads)


def _refine_job(args):
    from trifield.physics.refine import RefineConfig, Solver
    edir, items, out_dir = args
    md = pickle.loads((Path(edir) / "mesh_data.pkl").read_bytes())
    S = Solver(md, "cpu", RefineConfig())
    tf = Path(edir) / "trap_factors.pkl"
    if tf.exists():
        S.set_trap_factors(pickle.loads(tf.read_bytes()))
    rows = []
    for design, vg, vd, guess in items:
        t0 = time.process_time()
        try:
            r = S.refine(guess, vg, vd)
            name = f"{design}__vg{vg:+.3f}_vd{vd:.3f}.npy"
            if r.converged:
                y = np.stack([r.psi, np.where(md.semi, r.phin, 0.0), np.where(md.semi, r.phip, 0.0)], 1)
                np.save(Path(out_dir) / name, y.astype(np.float32))
            rows.append(dict(design=design, vg=vg, vd=vd, file=name if r.converged else None, conv=bool(r.converged),
                             merit=float(r.merit), newton=int(r.newton_iters), cpu=time.process_time() - t0))
        except Exception as exc:                                          # noqa: broad-except
            rows.append(dict(design=design, vg=vg, vd=vd, error=str(exc)))
    return rows


def pseudo_labels(ckpts, mesh_dir, export_roots, out, per_design=4, seed=0, exclude=(), workers=8, threads=2,
                  device="auto") -> list[dict]:
    """Refined targets for UnlabeledBiasDataset(mesh_dir, per_design, seed, exclude) items -> out/index.jsonl + .npy."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mpc
    from trifield.data.dataset import UnlabeledBiasDataset
    from trifield.inference import load_ensemble, predict_ensemble, resolve_device
    from trifield.physics.loss import PhysicsRegistry
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ds = UnlabeledBiasDataset(mesh_dir, per_design=per_design, seed=seed, exclude=exclude)
    reg = PhysicsRegistry(export_roots)
    models = load_ensemble(ckpts, device=resolve_device(device))
    jobs: dict = {}
    for i in range(len(ds)):
        g = ds[i]
        if g.sim_name not in reg:
            continue
        mean, _ = predict_ensemble(models, g)
        jobs.setdefault(str(reg.paths[g.sim_name]), []).append((g.sim_name, float(g.vg), float(g.vd), mean.double().numpy()))
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows = []
    with ProcessPoolExecutor(workers, mp_context=mpc.get_context("spawn"), initializer=_init, initargs=(threads,)) as ex, \
            open(out / "index.jsonl", "w") as fo:
        for fu in as_completed([ex.submit(_refine_job, (d, it, str(out))) for d, it in jobs.items()]):
            for r in fu.result():
                fo.write(json.dumps(r) + "\n"); rows.append(r)
            fo.flush()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="physics pseudo-labels for mesh-only designs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("graphs"); a1.add_argument("--pool", required=True); a1.add_argument("--out", required=True)
    a2 = sub.add_parser("labels")
    a2.add_argument("--ckpts", nargs="+", required=True)
    a2.add_argument("--mesh_dir", required=True)
    a2.add_argument("--exports", nargs="+", required=True, help="roots of <design>/twin/physics exports")
    a2.add_argument("--out", required=True)
    a2.add_argument("--per_design", type=int, default=4)
    a2.add_argument("--seed", type=int, default=0)
    a2.add_argument("--exclude", default="", help="file of design names to leave out (e.g. test designs)")
    a2.add_argument("--workers", type=int, default=8)
    a2.add_argument("--threads", type=int, default=2)
    a2.add_argument("--device", default="auto", help="surrogate inference device (refinement runs on the CPU)")
    a = ap.parse_args(argv)
    if a.cmd == "graphs":
        print(f"{build_mesh_graphs(Path(a.pool), Path(a.out))} mesh graphs written")
        return
    excl = [l.strip() for l in open(a.exclude)] if a.exclude else []
    rows = pseudo_labels(a.ckpts, a.mesh_dir, a.exports, a.out, a.per_design, a.seed, excl, a.workers, a.threads, a.device)
    conv = [r for r in rows if r.get("conv")]
    print(f"{len(conv)}/{len(rows)} converged; median CPU {np.median([r['cpu'] for r in rows if 'cpu' in r]):.1f} s")


if __name__ == "__main__":
    main()
