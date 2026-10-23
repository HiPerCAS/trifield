"""Per-mesh Sentaurus exports for the physics module (run once per design).

1. Box-method data: a 0-iteration sdevice run with ``Math{BoxMeasureFromFile BoxCoefficientsFromFile}``
   (no ``GrdNumbering``) in a directory without a debug file makes Sentaurus write
   ``MeasureCoefficientsDebug`` in its internal numbering. Internal units are a scaled length; the
   scale is calibrated from the total element volume. The element order equals the TDR bulk element
   order (checked: count + per-region volume conservation, ``MeshData.box_check``).
   NEVER run production decks in that directory: with the keywords absent the file is ignored,
   but a deck that keeps them would read it.
2. Constants: an equilibrium solve (Vg = Vd = 0) with extra Plot fields (affinity, effective band gap,
   N_C/N_V, doping, conduction band edge, mole fraction). Gives the per-vertex band parameters, the
   potential reference C0 = E_C + psi + chi, and the boundary constants (gate and Gold metal
   potentials, AlGaN(x=0) Gold-corner offset).
"""
from __future__ import annotations

import pickle
import re
import shutil
import subprocess
from pathlib import Path

CONST_PLOT = ("ElectronAffinity EffectiveBandGap eEffectiveStateDensity hEffectiveStateDensity "
              "DopingConcentration ConductionBandEnergy xMoleFraction SpaceCharge eDensity hDensity")
SDEVICE = ["sdevice", "-q", "-n", "--threads", "1"]


def _run(deck: str, tag: str, cwd: Path) -> None:
    (cwd / f"{tag}.cmd").write_text(deck)
    with open(cwd / f"{tag}.out", "w") as fo:
        subprocess.run(SDEVICE + [f"{tag}.cmd"], cwd=cwd, stdout=fo, stderr=subprocess.STDOUT, check=False)


def export_mesh_data(work: Path, base_deck: str, grid: str = "grid_des.tdr", dfise_grd: str = "msh.grd",
                     physics=None, force: bool = False):
    """Run both exports in ``work/physics`` (cached) and return a MeshData.

    work: the twin work directory containing ``grid`` (TDR the device is solved on), its DF-ISE
    round trip ``dfise_grd`` and ``parameter.par``. base_deck: the production sdevice deck."""
    from trifield.data.tdr_extract import convert_tdr_to_dfise
    from trifield.physics.mesh_data import MeshData
    from trifield.tcad.twin import EQUILIBRIUM, make_deck
    work = Path(work)
    pdir = work / "physics"
    cache = pdir / "mesh_data.pkl"
    if cache.exists() and not force:
        return pickle.loads(cache.read_bytes())
    if pdir.exists():
        shutil.rmtree(pdir)
    box, const = pdir / "box", pdir / "const"
    for d in (box, const):
        d.mkdir(parents=True)
        shutil.copy(work / grid, d)
        shutil.copy(work / "parameter.par", d)
    deck = re.sub(r"(^Plot\s*\{)", r"\1\n\t" + CONST_PLOT + "\n", base_deck, count=1, flags=re.M)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:                     # the two exports are independent sdevice runs
        jobs = [ex.submit(_run, make_deck(base_deck, "box", 0.0, 0.0, "\tCoupled(Iterations= 0){ Poisson }", grid=grid,
                                          math_extra="BoxMeasureFromFile BoxCoefficientsFromFile"), "box", box),
                ex.submit(_run, make_deck(deck, "const", 0.0, 0.0, EQUILIBRIUM, grid=grid), "const", const)]
        for j in jobs:
            j.result()
    if not (box / "MeasureCoefficientsDebug").exists():
        raise RuntimeError(f"box-coefficient export failed, see {box / 'box.out'}")
    if not (const / "const_des.tdr").exists():
        raise RuntimeError(f"constants export failed, see {const / 'const.out'}")
    _, dat = convert_tdr_to_dfise(const / "const_des.tdr", const, "const")
    md = MeshData(work / grid, work / dfise_grd, box / "MeasureCoefficientsDebug", dat, physics)
    cache.write_bytes(pickle.dumps(md))
    return md
