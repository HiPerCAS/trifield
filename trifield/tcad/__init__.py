"""Sentaurus TCAD wrapper: render command templates, run sde / sdevice,
and turn the resulting TDR snapshots into mesh graphs.

Requires a licensed Synopsys Sentaurus installation (``sde``, ``sdevice`` and
``tdx`` on PATH or under ``$SENTAURUS_BIN``) and a material parameter file,
see ``trifield/tcad/README.md``.
"""
from trifield.tcad.simulation import (SimulationParams, Simulation, MeshOnlySimulation,
                                      build_snapshot_graphs, PLTFile)

__all__ = ["SimulationParams", "Simulation", "MeshOnlySimulation",
           "build_snapshot_graphs", "PLTFile"]
