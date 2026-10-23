"""Train the surrogate.

    python -m trifield.train --data_dir data/train --val_data_dir data/val \
        --epochs 500 --run_name pignn --seed 0

Recipe of the paper: AdamW (lr 1e-3, weight decay 1e-4), cosine schedule,
gradient clipping at 5, dynamic batching by node count (~28k nodes / batch),
data loss + current-continuity residual whose weight ramps linearly from 0 to
``physics_max_weight`` between ``physics_start`` and ``physics_start +
physics_ramp`` epochs. ``--physics_max_weight 0`` gives the data-only
ablation. Every ``checkpoint_every`` epochs a checkpoint
``<out_dir>/<run_name>/epoch_NNNN_seedS.pt`` is written (none is deleted, so
the best epoch can be chosen afterwards); ``last.pt`` is always the final one.
Train several seeds to get an ensemble.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader, DynamicBatchSampler

from trifield.data.dataset import BaselineDeltaDataset, DeviceMeshDataset, HybridDataset
from trifield.losses import (CombinedLoss, channel_weights, gate_weights, ContinuityResidualLoss, PoissonResidualLoss,
                             contact_mask)
from trifield.model import DeviceGNN

log = logging.getLogger("trifield.train")
FIELDS = ("psi", "phi_n", "phi_p")


def compute_target_stats(ds) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-field mean / std over interior (non-contact) nodes of the training set."""
    n_fields = int(ds[0].y.shape[1])
    s = torch.zeros(n_fields, dtype=torch.float64)
    s2 = torch.zeros(n_fields, dtype=torch.float64)
    count = 0.0
    for i in range(len(ds)):
        d = ds[i]
        m = ~contact_mask(d.x)
        if not m.any():
            continue
        v = getattr(d, "y_valid", None)
        w = (v[m] if v is not None else torch.ones_like(d.y[m], dtype=torch.bool)).to(torch.float64)
        y = d.y[m].to(torch.float64)
        s += (y * w).sum(0)
        s2 += (y * y * w).sum(0)
        count = count + w.sum(0)
    count = torch.as_tensor(count, dtype=torch.float64).clamp(min=1.0)
    mean = s / count
    std = (s2 / count - mean * mean).clamp(min=1e-6).sqrt()
    return mean.float(), std.float()


@torch.no_grad()
def node_weights(batch, channel_weight: float, gate_weight: float):
    """Optional per-node data-loss weights (channel / depletion edge, gate stack); None = uniform."""
    w = None
    if channel_weight > 0:
        w = channel_weights(batch.y, batch.x, channel_weight)
    if gate_weight > 0:
        g = gate_weights(batch.x, batch.edge_index, gate_weight)
        w = g if w is None else w * g
    return w


@torch.no_grad()
def evaluate_val(model, ds, device, max_nodes_per_batch: int) -> dict:
    """Per-field RMSE [V] and mean continuity residual over a dataset."""
    was_training = model.training
    model.eval()
    loader = DataLoader(ds, batch_sampler=DynamicBatchSampler(
        ds, max_num=max_nodes_per_batch, mode="node", shuffle=False))
    n_fields = int(ds[0].y.shape[1])
    se = torch.zeros(n_fields, dtype=torch.float64, device=device)
    n_nodes = 0.0
    cont, pois, n_batches = 0.0, 0.0, 0
    cont_loss, pois_loss = ContinuityResidualLoss(), PoissonResidualLoss()
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        se += ((pred - batch.y) ** 2).sum(0).to(torch.float64)
        n_nodes += batch.num_nodes
        cm = contact_mask(batch.x)
        cont += float(cont_loss(pred, batch.edge_index, batch.edge_attr, cm))
        pois += float(pois_loss(pred, batch.x, batch.edge_index, batch.edge_attr, cm))
        n_batches += 1
    if was_training:
        model.train()
    rmse = (se / max(n_nodes, 1)).sqrt().cpu()
    return {"rmse": rmse, "continuity": cont / max(n_batches, 1), "poisson": pois / max(n_batches, 1)}


def save_checkpoint(path: Path, model, y_mean, y_std, epoch: int, seed: int, args: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": dict(model.config),
                "y_mean": y_mean, "y_std": y_std, "epoch": epoch, "seed": seed,
                "train_args": args}, path)
    return path


def train(data_dir, val_data_dir=None, epochs=500, lr=1e-3, hidden_dim=128, n_layers=8,
          max_nodes_per_batch=28000, physics_start=20, physics_ramp=150,
          physics_max_weight=0.1, cont_weight=0.1, poisson_weight=0.0,
          grad_checkpoint=False, seed=0, device="auto", run_name="run",
          out_dir="checkpoints", checkpoint_every=50, val_every=25,
          lr_scheduler="cosine", init_from=None, log_every=10, cache_data=False, global_node=False,
          baseline_delta=False, weight_decay=1e-4, act="leaky", norm="layer",
          input_norm=False, local_y=False, drop_vor=False, balance=False,
          channel_weight=0.0, gate_weight=0.0, edge_weight=0.0, hybrid_from=None, hybrid_x="",
          hybrid_edge="", hybrid_drop_valid=False):
    args = {k: v for k, v in locals().items()}
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)

    if hybrid_from:
        import functools
        DS = functools.partial(HybridDataset, other_dirs=hybrid_from.split(","),
                               x_cols=[int(c) for c in hybrid_x.split(",") if c],
                               edge_cols=[int(c) for c in hybrid_edge.split(",") if c],
                               drop_valid=hybrid_drop_valid)
    else:
        DS = BaselineDeltaDataset if baseline_delta else DeviceMeshDataset
    ds = DS(data_dir, cache=cache_data)
    if len(ds) == 0:
        raise SystemExit(f"no .pt graphs in {data_dir}")
    val_ds = DS(val_data_dir, cache=cache_data) if val_data_dir else None
    if val_ds is not None and len(val_ds) == 0:
        val_ds = None
    log.info("device=%s seed=%d train=%s val=%s", dev, seed, ds, val_ds)

    y_mean, y_std = compute_target_stats(ds)
    log.info("target stats: mean=%s std=%s", [round(v, 3) for v in y_mean.tolist()],
             [round(v, 3) for v in y_std.tolist()])
    y_mean_d, y_std_d = y_mean.to(dev), y_std.to(dev)

    loader = DataLoader(ds, batch_sampler=DynamicBatchSampler(
        ds, max_num=max_nodes_per_batch, mode="node", shuffle=True))

    sample = ds[0]
    model = DeviceGNN(in_dim=sample.x.shape[1], hidden_dim=hidden_dim,
                      out_dim=sample.y.shape[1], n_layers=n_layers,
                      edge_dim=sample.edge_attr.shape[1], grad_checkpoint=grad_checkpoint,
                      global_node=global_node, act=act, norm=norm, input_norm=input_norm,
                      local_y=local_y, drop_cols=(16,) if drop_vor else ()).to(dev)
    model.set_input_stats(ds)
    if init_from:
        model.load_state_dict(torch.load(init_from, map_location=dev, weights_only=False)["model_state"])
        log.info("warm start from %s", init_from)
    log.info("model: %s parameters", f"{model.num_parameters():,}")

    criterion = CombinedLoss(w_cont=cont_weight, w_pois=poisson_weight, balance=balance,
                             w_edge=edge_weight).to(dev)
    optimizer = torch.optim.AdamW([{"params": model.parameters()},
                                   {"params": criterion.parameters(), "weight_decay": 0.0}],
                                  lr=lr, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if lr_scheduler == "cosine" else None)

    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir / f"tb_seed{seed}"))
    except Exception:                                 # tensorboard is optional
        pass
    csv_path = run_dir / f"log_seed{seed}.csv"
    csv_file = csv_path.open("w", newline="")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(["epoch", "total", "data", "continuity", "poisson", "physics_weight", "lr",
                    "val_rmse_psi", "val_rmse_phi_n", "val_rmse_phi_p", "val_continuity", "val_poisson",
                    "sec"])

    model.train()
    epoch = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if epoch >= physics_start and physics_max_weight > 0:
            physics_weight = physics_max_weight * min(1.0, (epoch - physics_start) / max(physics_ramp, 1))
        else:
            physics_weight = 0.0
        sums = {"total": 0.0, "data": 0.0, "continuity": 0.0, "poisson": 0.0}
        n_batches = 0
        for batch in loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            pred_phys = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            losses = criterion(pred_norm=(pred_phys - y_mean_d) / y_std_d,
                               target_norm=(batch.y - y_mean_d) / y_std_d,
                               pred_phys=pred_phys, x=batch.x,
                               edge_index=batch.edge_index, edge_attr=batch.edge_attr,
                               contact=contact_mask(batch.x), physics_weight=physics_weight,
                               weights=node_weights(batch, channel_weight, gate_weight),
                               valid=getattr(batch, "y_valid", None))
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for k in sums:
                sums[k] += float(losses[k].detach())
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        n = max(n_batches, 1)
        avg = {k: v / n for k, v in sums.items()}

        vm = None
        if val_ds is not None and val_every > 0 and epoch % val_every == 0:
            vm = evaluate_val(model, val_ds, dev, max_nodes_per_batch)
        if epoch % log_every == 0 or epoch == 1 or vm is not None:
            msg = (f"epoch {epoch:4d} | total={avg['total']:.5f} data={avg['data']:.5f} "
                   f"cont={avg['continuity']:.4f} pw={physics_weight:.3f} lr={cur_lr:.2e} "
                   f"({time.time() - t0:.0f}s)")
            if vm is not None:
                r = vm["rmse"]
                msg += (f" | val RMSE[V] psi={r[0]:.3f} phi_n={r[1]:.3f} phi_p={r[2]:.3f} "
                        f"cont={vm['continuity']:.3f} pois={vm['poisson']:.3f}")
            log.info(msg)
        if writer is not None:
            for k, v in avg.items():
                writer.add_scalar(f"loss/{k}", v, epoch)
            writer.add_scalar("physics_weight", physics_weight, epoch)
            writer.add_scalar("lr", cur_lr, epoch)
            if vm is not None:
                for i, f in enumerate(FIELDS):
                    writer.add_scalar(f"val/rmse_{f}", float(vm["rmse"][i]), epoch)
                writer.add_scalar("val/continuity", vm["continuity"], epoch)
                writer.add_scalar("val/poisson", vm["poisson"], epoch)
        csv_w.writerow([epoch, avg["total"], avg["data"], avg["continuity"], avg["poisson"],
                        physics_weight, cur_lr]
                       + ([float(v) for v in vm["rmse"]] + [vm["continuity"], vm["poisson"]]
                          if vm is not None else [None] * 5) + [round(time.time() - t0, 1)])
        csv_file.flush()
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}_seed{seed}.pt", model, y_mean, y_std,
                            epoch, seed, args)
    final = save_checkpoint(run_dir / f"last_seed{seed}.pt", model, y_mean, y_std, epoch, seed, args)
    csv_file.close()
    if writer is not None:
        writer.close()
    log.info("done: %s", final)
    return final


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the mesh-native surrogate")
    ap.add_argument("--data_dir", required=True, help="directory of training graphs (*.pt)")
    ap.add_argument("--val_data_dir", default=None, help="held-out graphs for validation")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--max_nodes_per_batch", type=int, default=28000,
                    help="dynamic-batching cap on the total node count per batch")
    ap.add_argument("--physics_start", type=int, default=20, help="epoch at which the physics weight starts ramping")
    ap.add_argument("--physics_ramp", type=int, default=150, help="epochs over which it ramps 0 -> max")
    ap.add_argument("--physics_max_weight", type=float, default=0.1, help="lambda_max (0 = data-only ablation)")
    ap.add_argument("--cont_weight", type=float, default=0.1, help="inner weight of the continuity residual")
    ap.add_argument("--poisson_weight", type=float, default=0.0, help="inner weight of the Poisson residual (0 = off)")
    ap.add_argument("--grad_checkpoint", action="store_true", help="trade compute for activation memory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run_name", default="run")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--checkpoint_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--lr_scheduler", choices=["cosine", "constant"], default="cosine")
    ap.add_argument("--init_from", default=None, help="warm-start weights from a checkpoint")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--baseline_delta", action="store_true",
                    help="learn the change from the same design's Vd = 0 solution (given as input)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--act", choices=["leaky", "silu", "gelu"], default="leaky")
    ap.add_argument("--norm", choices=["layer", "none"], default="layer")
    ap.add_argument("--input_norm", action="store_true", help="standardise node / edge features")
    ap.add_argument("--local_y", action="store_true", help="fin-local y coordinate instead of fin index")
    ap.add_argument("--drop_vor", action="store_true", help="zero the near-constant log-volume feature")
    ap.add_argument("--balance", action="store_true", help="homoscedastic per (material, field) data loss")
    ap.add_argument("--hybrid_from", default=None, help="ablation: comma-separated dirs with older graphs")
    ap.add_argument("--hybrid_x", default="", help="x columns to take from the older graphs, e.g. 3,4,5,6,7")
    ap.add_argument("--hybrid_edge", default="", help="edge_attr columns to take from the older graphs")
    ap.add_argument("--hybrid_drop_valid", action="store_true", help="ablation: ignore y_valid")
    ap.add_argument("--channel_weight", type=float, default=0.0,
                    help="extra weight on conducting-channel / depletion-edge nodes (needs --balance)")
    ap.add_argument("--gate_weight", type=float, default=0.0,
                    help="extra data-loss weight on semiconductor nodes within 2 hops of the gate oxide (8 recommended)")
    ap.add_argument("--edge_weight", type=float, default=0.0,
                    help="weight of the edge-difference (local gradient) loss on semiconductor edges")
    ap.add_argument("--global_node", action="store_true",
                    help="learned virtual node after every block (long-range electrostatics)")
    ap.add_argument("--cache_data", action="store_true", help="keep all graphs in RAM after first load")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    train(**vars(args))


if __name__ == "__main__":
    main()
