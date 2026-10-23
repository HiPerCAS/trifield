"""Load checkpoints and run (ensemble) inference on mesh graphs."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from trifield.data.dataset import load_graph
from trifield.losses import contact_mask
from trifield.model import DeviceGNN

__all__ = ["resolve_device", "load_model", "load_ensemble", "predict",
           "predict_ensemble", "interior_mask", "ensemble_uncertainty", "load_graph"]


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(path, device: str | torch.device = "auto") -> tuple[DeviceGNN, dict]:
    """Load a checkpoint written by ``trifield.train``.

    Returns the model in eval mode and a metadata dict with the target
    normalisation statistics (``y_mean``/``y_std``), ``config``, ``epoch``,
    ``seed`` and ``path``. Predictions are in physical units (volts); the
    statistics are only needed to continue training.
    """
    dev = resolve_device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    model = DeviceGNN.from_config(ckpt["config"]).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {"y_mean": ckpt.get("y_mean"), "y_std": ckpt.get("y_std"),
            "config": ckpt["config"], "epoch": ckpt.get("epoch"),
            "seed": ckpt.get("seed"), "path": str(path), "device": dev}
    return model, meta


def load_ensemble(paths: Sequence, device: str | torch.device = "auto") -> list[DeviceGNN]:
    """Load independently seeded members; ``paths`` may contain glob patterns."""
    files: list[Path] = []
    for p in paths:
        p = str(p)
        matches = sorted(Path().glob(p)) if any(c in p for c in "*?[") else [Path(p)]
        files.extend(matches)
    if not files:
        raise FileNotFoundError(f"no checkpoints match {list(paths)}")
    return [load_model(f, device)[0] for f in files]


@torch.no_grad()
def predict(model: DeviceGNN, data, device: str | torch.device | None = None) -> torch.Tensor:
    """Per-node (psi, Phi_n, Phi_p) in volts for one graph, returned on CPU."""
    dev = next(model.parameters()).device if device is None else resolve_device(device)
    x, ei, ea = data.x.to(dev), data.edge_index.to(dev), data.edge_attr.to(dev)
    return model(x, ei, ea).cpu()


@torch.no_grad()
def predict_ensemble(models: Sequence[DeviceGNN], data) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensemble mean (N, 3) and per-node disagreement sigma_v (N,) in volts,
    sigma_v^2 = mean_j || f_j(v) - fbar(v) ||^2 over the members."""
    preds = torch.stack([predict(m, data) for m in models])          # (M, N, 3)
    mean = preds.mean(dim=0)
    sigma = ((preds - mean) ** 2).sum(dim=-1).mean(dim=0).sqrt()
    return mean, sigma


def interior_mask(data) -> torch.Tensor:
    """Nodes that are not source/drain/gate contacts (the loss / uncertainty mask)."""
    return ~contact_mask(data.x)


def ensemble_uncertainty(sigma: torch.Tensor, data) -> float:
    """Snapshot-level uncertainty u_k = sqrt(mean over interior nodes of sigma_v^2)."""
    m = interior_mask(data)
    s = sigma[m] if m.any() else sigma
    return float(torch.sqrt((s ** 2).mean()))
