"""trifield: mesh-native physics-informed graph surrogates for TCAD.

The surrogate predicts the three drift-diffusion unknowns (electrostatic
potential psi, electron and hole quasi-Fermi potentials Phi_n / Phi_p) at every
node of a Sentaurus TCAD mesh, and is trained with a data loss plus a
finite-volume current-continuity residual. See README.md.
"""
from trifield.params import FIXED, PARAM_ORDER, RANGES, SFIN, BIAS_SNAPSHOTS
from trifield.model import DeviceGNN
from trifield.inference import (load_model, load_ensemble, predict,
                                predict_ensemble, interior_mask, load_graph)
from trifield.device_metrics import DeviceMetrics, extract_device_metrics

__version__ = "1.0.0"

__all__ = [
    "FIXED", "PARAM_ORDER", "RANGES", "SFIN", "BIAS_SNAPSHOTS",
    "DeviceGNN", "load_model", "load_ensemble", "predict", "predict_ensemble",
    "interior_mask", "load_graph", "DeviceMetrics", "extract_device_metrics",
]
