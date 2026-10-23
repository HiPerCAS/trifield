"""Field-level error metrics (physical units, volts)."""
from __future__ import annotations

import torch

FIELDS = ("psi", "phi_n", "phi_p")


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (torch.sqrt(torch.mean((pred - true) ** 2))
            / torch.sqrt(torch.mean(true ** 2) + 1e-30)).item()


def max_abs_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.max(torch.abs(pred - true)).item()


def field_errors(pred: torch.Tensor, true: torch.Tensor,
                 mask: torch.Tensor | None = None) -> dict:
    """Per-field RMSE / relative L2 / max abs error, optionally over ``mask`` nodes."""
    if mask is not None:
        pred, true = pred[mask], true[mask]
    out = {}
    for i, name in enumerate(FIELDS[: pred.shape[1]]):
        out[name] = {"rmse": rmse(pred[:, i], true[:, i]),
                     "rel_l2": relative_l2(pred[:, i], true[:, i]),
                     "max_ae": max_abs_error(pred[:, i], true[:, i])}
    return out


@torch.no_grad()
def continuity_residual(pred_phys: torch.Tensor, data) -> float:
    """Current-continuity residual of a field under the finite-volume operator
    of the training loss (``trifield.losses.ContinuityResidualLoss``).

    The operator is a simplification of the solver's scheme (unit face areas,
    Boltzmann densities, constant mobility), so the simulated fields themselves
    do not score zero: always read a model's residual next to the residual of
    the TCAD fields on the same graph (``continuity_residual(data.y, data)``).
    """
    from trifield.losses import ContinuityResidualLoss, contact_mask
    return float(ContinuityResidualLoss()(pred_phys, data.edge_index, data.edge_attr,
                                          contact_mask(data.x)))
