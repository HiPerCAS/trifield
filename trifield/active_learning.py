"""Four-criterion data acquisition for the TCAD-in-the-loop digital twin.

Which designs should be simulated next so the twin gets faster and stays
accurate where it matters? Every candidate design in a pool of mesh-only
graphs (Sentaurus SDE only, ~30 s each, no solve) is scored on four criteria,
each z-scored across the pool:

1. **uncertainty**  U(x): deep-ensemble disagreement, the interior RMS of the
   per-node spread sigma_v, worst case over the bias points the twin solves at.
2. **novelty**      D(x): distance of the design to the training set in the
   normalised design space [0,1]^6 (+ fin count), i.e. how unlike anything the
   surrogate has seen it is (1 - similarity).
3. **objective**    O(x): how good the design is. With ``--objectives twin`` the
   objectives come from the warm-start twin itself (Sentaurus-accurate Id,max
   and Vth, run on a shortlist); otherwise from the fast closed-form proxies.
4. **runtime**      R(x): how much the warm start is expected to struggle,
   from the physics residuals (current continuity + Poisson) of the predicted
   fields and the mesh size. A large residual means the prediction is far from
   a solution of the discrete equations, so the warm-started Newton solve
   needs more iterations; simulating such a design teaches the surrogate
   exactly where the twin is slow. ``--runtime_sign -1`` instead penalises
   expensive designs (budget-limited campaigns).

    score = w_u z(U) + w_d z(D) + w_o z(O) + w_r z(R)

followed by a greedy top-K selection with a diversity filter so the batch
does not collapse onto one region.

    python -m trifield.active_learning --ckpts ckpt/*.pt --candidates runs/candidates \
        --train_designs data/train --top_k 16 --out_csv runs/selection.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from trifield.data.dataset import group_by_design, load_graph
from trifield.device_metrics import extract_device_metrics
from trifield.inference import ensemble_uncertainty, load_ensemble, predict_ensemble
from trifield.losses import ContinuityResidualLoss, PoissonResidualLoss, contact_mask
from trifield.params import PARAM_ORDER, design_vector

log = logging.getLogger("trifield.active_learning")

TWIN_BIASES = ((5.0, 12.0), (5.0, 3.0))       # bias points the twin solves at directly
FIN_SCALE = 0.25                               # weight of one fin in the novelty distance


def _vec(params: dict) -> np.ndarray:
    return np.concatenate([design_vector(params), [FIN_SCALE * float(params.get("Nfins", 1))]])


def training_design_vectors(train_dir) -> np.ndarray:
    """One vector per training design (read from the first graph of each design)."""
    vecs = []
    for files in group_by_design(Path(train_dir).glob("*.pt")).values():
        p = getattr(load_graph(files[0]), "params", None)
        if p:
            vecs.append(_vec(p))
    return np.asarray(vecs)


def novelty(params: dict, train_vecs: np.ndarray) -> float:
    if train_vecs is None or len(train_vecs) == 0:
        return 1.0
    return float(np.min(np.linalg.norm(train_vecs - _vec(params), axis=1)))


@torch.no_grad()
def score_candidate(models: Sequence, graphs: Sequence, train_vecs=None) -> dict:
    """Uncertainty, residuals, proxies and novelty of one candidate from its graphs."""
    cont_loss, pois_loss = ContinuityResidualLoss(), PoissonResidualLoss()
    unc, unc_twin, cont, pois, snaps = [], [], [], [], []
    n_nodes = 0
    for g in graphs:
        data = load_graph(g) if not hasattr(g, "x") else g
        mean, sigma = predict_ensemble(models, data)
        u = ensemble_uncertainty(sigma, data)
        unc.append(u)
        bias = (round(float(data.vg), 2), round(float(data.vd), 2))
        if bias in TWIN_BIASES:
            unc_twin.append(u)
            cm = contact_mask(data.x)
            cont.append(float(cont_loss(mean, data.edge_index, data.edge_attr, cm)))
            pois.append(float(pois_loss(mean, data.x, data.edge_index, data.edge_attr, cm)))
        snaps.append((data, mean))
        n_nodes = int(data.num_nodes)
    m = extract_device_metrics(snaps)
    params = dict(getattr(snaps[0][0], "params", None) or {})
    return {
        "params": params, "nfins": int(params.get("Nfins", getattr(snaps[0][0], "nfins", 1))),
        "n_nodes": n_nodes,
        "uncertainty": max(unc_twin or unc),
        "novelty": novelty(params, train_vecs) if params else 0.0,
        "residual_cont": float(np.mean(cont)) if cont else 0.0,
        "residual_pois": float(np.mean(pois)) if pois else 0.0,
        "id_proxy": float(m.id_max or 0.0), "vth_proxy": float(m.vth or 0.0),
    }


def _z(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-30 else 1.0)


def combine(rows: list[dict], w_u=1.0, w_d=1.0, w_o=0.5, w_r=0.5, runtime_sign=1.0,
            objective_key: str = "proxy") -> list[dict]:
    """Add per-criterion z-scores and the total score; sort descending."""
    if not rows:
        return rows
    if objective_key == "twin":
        idm = [r.get("id_max_twin") or 0.0 for r in rows]
        vth = [r.get("vth_twin") if r.get("vth_twin") is not None else -10.0 for r in rows]
    else:
        idm = [np.log10(max(r["id_proxy"], 1e-30)) for r in rows]
        vth = [r["vth_proxy"] for r in rows]
    obj = _z(idm) + _z(vth)
    # runtime: residuals of the predicted fields (log scale) and the mesh size
    rt = _z(np.log10(np.asarray([r["residual_cont"] for r in rows]) + 1e-12)) \
        + _z(np.log10(np.asarray([r["residual_pois"] for r in rows]) + 1e-12)) \
        + _z(np.log10([r["n_nodes"] for r in rows]))
    for r, u, d, o, t in zip(rows, _z([r["uncertainty"] for r in rows]),
                             _z([r["novelty"] for r in rows]), _z(obj), _z(rt)):
        r.update(z_uncertainty=float(u), z_novelty=float(d), z_objective=float(o), z_runtime=float(t))
        r["score"] = w_u * u + w_d * d + w_o * o + w_r * runtime_sign * t
    rows.sort(key=lambda r: -r["score"])
    return rows


def nondominated_fronts(F: np.ndarray) -> list[np.ndarray]:
    """Fast non-dominated sorting (NSGA-II) of objectives F (n, m), all minimised."""
    n = len(F)
    dominates = [np.where(np.all(F[i] <= F, 1) & np.any(F[i] < F, 1))[0] for i in range(n)]
    count = np.array([int((np.all(F <= F[i], 1) & np.any(F < F[i], 1)).sum()) for i in range(n)])
    fronts, cur = [], list(np.where(count == 0)[0])
    while cur:
        fronts.append(np.array(cur))
        nxt = []
        for i in cur:
            for j in dominates[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    for k in range(m):
        o = np.argsort(F[:, k])
        d[o[0]] = d[o[-1]] = np.inf
        span = F[o[-1], k] - F[o[0], k]
        if span > 0 and n > 2:
            d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / span
    return d


def pareto_rank(rows: list[dict]) -> list[dict]:
    """NSGA-II ordering over (information = U + D, objective, -labelling cost): front by front, each
    front by decreasing crowding distance. Labelling cost is log mesh size (the twin's labelling time
    grows ~ n^0.5; residuals and uncertainty did not predict it)."""
    if not rows:
        return rows
    F = np.c_[-(np.asarray([r["z_uncertainty"] for r in rows]) + np.asarray([r["z_novelty"] for r in rows])),
              -np.asarray([r["z_objective"] for r in rows]),
              np.log10([r["n_nodes"] for r in rows])]
    order = []
    for k, fr in enumerate(nondominated_fronts(F)):
        cd = crowding_distance(F[fr])
        for i in fr[np.argsort(-cd)]:
            rows[i]["front"] = k
            order.append(i)
    return [rows[i] for i in order]


def select_diverse(scored: list[dict], top_k: int, threshold: float = 0.15) -> list[dict]:
    """Greedy top-K, rejecting candidates within ``threshold`` (normalised design
    space, fin count included) of an already selected one."""
    selected: list[dict] = []
    for c in scored:
        if len(selected) >= top_k:
            break
        if c["params"] and any(np.linalg.norm(_vec(c["params"]) - _vec(s["params"])) < threshold
                               for s in selected if s["params"]):
            continue
        selected.append(c)
    return selected


def twin_objectives(rows: list[dict], candidates_dir: Path, models, par, workers: int = 4) -> None:
    """Sentaurus-accurate objectives for a shortlist via the warm-start twin (in place).
    The candidates' meshes must be kept (``campaign --mesh_only --keep``)."""
    from concurrent.futures import ProcessPoolExecutor
    from trifield.tcad.twin import Twin
    for r in rows:                       # sequential per design; each twin runs 3 solves in parallel
        run_dir = Path(candidates_dir) / r["name"]
        if not (run_dir / "node_msh.tdr").exists():
            continue
        res = Twin(r["params"], run_dir, models, par, "warm").run()
        r["id_max_twin"], r["vth_twin"], r["ron_twin"] = res["id_max"], res["vth"], res["ron"]
        r["twin_wall"] = res["wall_without_mesh"]


def write_csv(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["name", "score", "z_uncertainty", "z_novelty", "z_objective", "z_runtime", "uncertainty",
            "novelty", "residual_cont", "residual_pois", "n_nodes", "id_proxy", "vth_proxy",
            "id_max_twin", "vth_twin", "ron_twin", "twin_wall", "nfins"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + PARAM_ORDER)
        for r in rows:
            w.writerow([r.get(c) for c in cols] + [r["params"].get(k) for k in PARAM_ORDER])


def main(argv=None):
    ap = argparse.ArgumentParser(description="four-criterion acquisition over a candidate pool")
    ap.add_argument("--ckpts", nargs="+", required=True, help="ensemble members (>= 2)")
    ap.add_argument("--candidates", required=True, help="directory of <cand>__<snapshot>.pt graphs")
    ap.add_argument("--train_designs", default=None, help="training graph directory (for novelty)")
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--strategy", choices=["pareto", "scalar"], default="pareto",
                    help="pareto: NSGA-II ranking of (uncertainty+novelty, objective, -mesh size); "
                         "scalar: weighted z-sum (weights below)")
    ap.add_argument("--w_uncertainty", type=float, default=1.0)
    ap.add_argument("--w_novelty", type=float, default=1.0)
    ap.add_argument("--w_objective", type=float, default=0.5)
    ap.add_argument("--w_runtime", type=float, default=0.5)
    ap.add_argument("--runtime_sign", type=float, default=1.0,
                    help="+1: prefer designs where the warm start will struggle; -1: penalise cost")
    ap.add_argument("--objectives", choices=["proxy", "twin"], default="proxy",
                    help="'twin': Sentaurus-accurate objectives via warm starts on a 3*top_k shortlist")
    ap.add_argument("--candidate_runs", default=None, help="kept mesh-only run dirs (for --objectives twin)")
    ap.add_argument("--par", default=None)
    ap.add_argument("--diversity", type=float, default=0.15)
    ap.add_argument("--out_csv", default="selection.csv")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    models = load_ensemble(args.ckpts, args.device)
    if len(models) < 2:
        raise SystemExit("ensemble uncertainty needs at least two checkpoints")
    train_vecs = training_design_vectors(args.train_designs) if args.train_designs else None
    pool = group_by_design(Path(args.candidates).glob("*.pt"))
    graphs = {k: [f for f in v if True] for k, v in pool.items()}
    log.info("scoring %d candidates with M=%d members", len(graphs), len(models))
    rows = []
    for name, files in graphs.items():
        try:
            r = score_candidate(models, files, train_vecs)
        except Exception as e:                        # noqa: broad-except
            log.warning("candidate %s failed: %s", name, e)
            continue
        r["name"] = name
        rows.append(r)
    weights = dict(w_u=args.w_uncertainty, w_d=args.w_novelty, w_o=args.w_objective,
                   w_r=args.w_runtime, runtime_sign=args.runtime_sign)
    rows = combine(rows, **weights)
    if args.objectives == "twin":
        short = rows[: 3 * args.top_k]
        twin_objectives(short, Path(args.candidate_runs or args.candidates), models, args.par)
        rows = combine(short, objective_key="twin", **weights)
    if args.strategy == "pareto":
        rows = pareto_rank(rows)
    selected = select_diverse(rows, args.top_k, args.diversity)
    write_csv(rows, Path(args.out_csv).with_name(Path(args.out_csv).stem + "_all.csv"))
    write_csv(selected, args.out_csv)
    for r in selected:
        log.info("%-30s score=%+.2f  U=%+.2f D=%+.2f O=%+.2f R=%+.2f  fins=%d", r["name"], r["score"],
                 r["z_uncertainty"], r["z_novelty"], r["z_objective"], r["z_runtime"], r["nfins"])
    log.info("selected %d/%d -> %s", len(selected), len(rows), args.out_csv)


if __name__ == "__main__":
    main()
