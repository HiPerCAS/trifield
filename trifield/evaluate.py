"""Evaluate checkpoints on a directory of graphs.

    python -m trifield.evaluate --ckpts models/pignn_ens_seed0.pt models/pignn_ens_seed1.pt \
        --data_dir data/val [--foms]

Reports per-field RMSE / relative L2 / max abs error (volts, all nodes) overall
and per fin count, the mean continuity residual of the predicted fields, and
with ``--foms`` the closed-form device metrics against those read from the
ground-truth fields (see the caveat in ``trifield.device_metrics``). With
several checkpoints the ensemble mean is evaluated.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from trifield.data.dataset import DeviceMeshDataset, design_name
from trifield.device_metrics import extract_device_metrics
from trifield.inference import load_ensemble, predict_ensemble
from trifield.metrics import FIELDS, continuity_residual, field_errors

log = logging.getLogger("trifield.evaluate")


def evaluate(ckpts, data_dir, device="auto", foms=False, pattern="*.pt") -> dict:
    models = load_ensemble(ckpts, device)
    ds = DeviceMeshDataset(data_dir, pattern)
    if len(ds) == 0:
        raise SystemExit(f"no graphs in {data_dir}")
    log.info("%d members, %s", len(models), ds)

    per_sample, by_design = [], defaultdict(list)
    for i in range(len(ds)):
        data = ds[i]
        mean, _ = predict_ensemble(models, data)
        errs = field_errors(mean, data.y)
        row = {"file": ds.files[i].name, "nfins": int(getattr(data, "nfins", 1)),
               "n_nodes": int(data.num_nodes), "continuity": continuity_residual(mean, data),
               "continuity_tcad": continuity_residual(data.y, data),
               **{f"{f}_{k}": v for f, d in errs.items() for k, v in d.items()}}
        per_sample.append(row)
        by_design[design_name(ds.files[i])].append((data, mean))

    def summarise(rows, label):
        out = {}
        for f in FIELDS:
            se = sum(r[f"{f}_rmse"] ** 2 * r["n_nodes"] for r in rows)
            nn = sum(r["n_nodes"] for r in rows)
            out[f] = {"rmse": float(np.sqrt(se / nn)),
                      "max_ae": float(max(r[f"{f}_max_ae"] for r in rows))}
        out["continuity"] = float(np.mean([r["continuity"] for r in rows]))
        out["continuity_tcad"] = float(np.mean([r["continuity_tcad"] for r in rows]))
        # RMSE in volts is the paper's metric: Phi_n / Phi_p vanish over whole
        # regions (and everywhere at Vd = 0), so a relative L2 is ill-defined.
        log.info("%-14s n=%4d | RMSE[V] psi=%.3f phi_n=%.3f phi_p=%.3f "
                 "| max|err|[V] %.2f/%.2f/%.2f | continuity residual %.3f (TCAD fields: %.3f)",
                 label, len(rows),
                 out["psi"]["rmse"], out["phi_n"]["rmse"], out["phi_p"]["rmse"],
                 out["psi"]["max_ae"], out["phi_n"]["max_ae"], out["phi_p"]["max_ae"],
                 out["continuity"], out["continuity_tcad"])
        return out

    result = {"all": summarise(per_sample, "all snapshots"), "by_nfins": {}}
    for nf in sorted({r["nfins"] for r in per_sample}):
        result["by_nfins"][nf] = summarise([r for r in per_sample if r["nfins"] == nf], f"N_fins={nf}")

    if foms:
        log.info("closed-form device-metric proxies, predicted vs ground-truth fields (per design; "
                 "vth in V, the others in relative units, see trifield.device_metrics):")
        errs = defaultdict(list)
        for name, snaps in sorted(by_design.items()):
            if len(snaps) < 2:
                continue
            pm = extract_device_metrics(snaps)
            gm = extract_device_metrics([(d, d.y) for d, _ in snaps])
            parts = []
            for k in ("id_max", "Ron", "gm", "vth"):
                pv, gv = getattr(pm, k), getattr(gm, k)
                if pv is not None and gv is not None and abs(gv) > 1e-30:
                    e = abs(pv - gv) / abs(gv) * 100
                    errs[k].append(e)
                    parts.append(f"{k}: pred={pv:.4g} true={gv:.4g} ({e:.0f}%)")
            log.info("  %s | %s", name, "  ".join(parts))
        result["fom_median_relerr_pct"] = {k: float(np.median(v)) for k, v in errs.items()}
        log.info("median relative error [%%]: %s",
                 {k: round(v, 1) for k, v in result["fom_median_relerr_pct"].items()})
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate the surrogate on a graph directory")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pattern", default="*.pt", help="glob under data_dir (e.g. '**/*.pt')")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--foms", action="store_true", help="also compare closed-form device metrics")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    evaluate(args.ckpts, args.data_dir, args.device, args.foms, args.pattern)


if __name__ == "__main__":
    main()
