"""Exact Sentaurus drift-diffusion residuals and a physics refinement of surrogate guesses.

    from trifield.physics import export_mesh_data, Solver
    md = export_mesh_data(twin_work_dir, twin.base)          # once per design (two short sdevice runs)
    res = Solver(md, device="cpu").refine(guess, vg, vd)     # guess: (n,3) psi, Phi_n, Phi_p
See README.md in this directory.
"""
from trifield.physics.export import export_mesh_data
from trifield.physics.mesh_data import DeckPhysics, MeshData, TrapSet
from trifield.physics.refine import RefineConfig, RefineResult, Solver

__all__ = ["export_mesh_data", "DeckPhysics", "MeshData", "TrapSet", "RefineConfig", "RefineResult", "Solver"]
