"""Warm-start digital twin: surrogate fields as the initial guess of Sentaurus Device.

The surrogate does not replace the drift-diffusion solve, it seeds it. For a
design the twin

1. meshes it with ``sde`` (the same structure script as the campaign),
2. predicts (psi, Phi_n, Phi_p) at every mesh node for each bias point it needs,
3. writes the prediction into a Sentaurus solution file (TDR) on the original
   mesh, and
4. runs ``sdevice`` with the production physics directly at that bias point,
   ``Load``-ing the prediction as the initial guess (no voltage ramp).

The figures of merit therefore come from the simulator itself, so their
accuracy is the simulator's; what the surrogate buys is time: a warm Newton
solve at the target bias instead of the long quasistationary ramps of the
production decks. The objectives are

* ``id_max``  drain current at (Vg = Vg11, Vd = 12 V)          [mA/mm]
* ``ron``     Vd / Id at (Vg = Vg11, Vd = 3 V)                   [Ohm mm]
* ``vth``     gate bias where Id = 1 mA/mm at Vd = 12 V          [V], found by a
               bracketed secant search on log10(Id) in which every point is a
               warm-started solve.

``mode="cold"`` runs the same algorithm without the surrogate (each point is
reached by a direct quasistationary ramp from equilibrium); it isolates what
the warm start contributes beyond the leaner solve recipe. The production
pipeline (``trifield.tcad.simulation.Simulation``) is the reference for both
accuracy and runtime.

    python -m trifield.tcad.twin --ckpts ckpt/*.pt --design examples/design.json --par parameter.par
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from trifield.params import FIXED, SFIN, area_factor
from trifield.tcad.simulation import (SDE_CMD, SDE_TEMPLATE, TEMPLATE_DIR, PLTFile,
                                      SimulationParams, find_parameter_file, render_template,
                                      run_timed)

log = logging.getLogger("trifield.tcad.twin")

FIELD_FUNCS = ("ElectrostaticPotential", "eQuasiFermiPotential", "hQuasiFermiPotential")
SDEVICE = ["sdevice", "-q", "-n", "--threads", "1"]
VTH_TARGET = 1.0            # mA/mm, same criterion as the production find_vth
VD_HIGH, VD_LIN = 12.0, 3.0
GRID = "grid_des.tdr"        # DF-ISE round-tripped mesh the twin solves on


# ---------------------------------------------------------------------------
# Physics anchoring of the surrogate prediction
# ---------------------------------------------------------------------------

def anchor_quasi_fermi(pred, data, quantile: float = 0.005):
    """Enforce what the boundary conditions already tell us about the quasi-Fermi levels.

    In the semiconductor, Phi_n and Phi_p obey a maximum principle: they lie between
    the contact biases (0 V at the source, Vd at the drain) and are exactly 0 at
    Vd = 0 (equilibrium). The current-continuity residual only sees their gradients,
    so a surrogate can carry a constant offset that it cannot see but Newton can:
    0.8 V of Phi_n offset is a e^(0.8/V_T) ~ 1e13 error in the electron density.
    Phi_n is mapped affinely onto [0, Vd] (robust quantiles, clipped); Phi_p is only
    zeroed at equilibrium. Nodes outside GaN/AlGaN are left alone (the quasi-Fermi
    levels are undefined there and are not written into the solver guess).
    """
    import torch
    p = pred.clone()
    semi = (data.x[:, 3] + data.x[:, 4]) > 0
    vd = float(data.vd)
    if not semi.any():
        return p
    if abs(vd) < 1e-9:
        p[semi, 1] = 0.0
        p[semi, 2] = 0.0
        return p
    v = p[semi, 1]
    lo, hi = torch.quantile(v, quantile), torch.quantile(v, 1 - quantile)
    p[semi, 1] = ((v - lo) / (hi - lo).clamp(min=1e-6) * vd).clamp(min(0.0, vd), max(0.0, vd))
    return p


CONDUCTORS = ("Gold", "TiN", "Aluminum", "Copper", "Metal", "PolySi")


def electrode_fermi_levels(mesh, vg: float, vd: float, vs: float = 0.0):
    """Per-vertex ``QuasiFermiPotential`` that Sentaurus keeps for every region:
    the applied bias of the electrode whose metal a vertex belongs to, 0 elsewhere.

    It is fully determined by the geometry and the bias, but it must be in the
    initial guess: leaving it at a zero-bias value (as a template written at
    equilibrium has) costs ~30 Newton iterations even when psi / Phi_n / Phi_p
    are exact, while setting it makes the exact solution converge in one.
    Each metal region is assigned to the contact (S / D / G) whose sheet touches it.
    """
    bias = {"S": vs, "D": vd, "G": vg}
    contact_verts = {c: set(map(int, mesh.region_vertex_indices.get(c, []))) for c in bias}
    qf = np.zeros(mesh.n_vertices)
    for reg, mat in zip(mesh.region_names, mesh.region_materials):
        if mat not in CONDUCTORS or reg not in mesh.region_vertex_indices:
            continue
        idx = mesh.region_vertex_indices[reg]
        verts = set(map(int, idx))
        hits = {c: len(verts & cv) for c, cv in contact_verts.items()}
        c = max(hits, key=hits.get)
        if hits[c] > 0:
            qf[idx] = bias[c]
    return qf


# ---------------------------------------------------------------------------
# DF-ISE helpers
# ---------------------------------------------------------------------------

def _match_brace(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        depth += (s[j] == "{") - (s[j] == "}")
        if depth == 0:
            return j
    return -1


def _rewrite_dat(src: Path, dst: Path, mesh=None, fields: dict | None = None) -> None:
    """Copy a DF-ISE .dat, dropping datasets whose name contains '(' (they break
    ``tdx --dfise2tdr``) and, if ``fields`` is given, replacing the values of those
    datasets with ``fields[name][region_vertex_indices[region]]``."""
    txt = src.read_text(errors="replace")
    im = re.search(r"Info\s*\{", txt)
    info = txt[im.end():_match_brace(txt, im.end() - 1)]
    dm = re.search(r"Data\s*\{", txt)
    data = txt[dm.end():_match_brace(txt, dm.end() - 1)]
    keep, names, funcs = [], [], []
    for m in re.finditer(r'Dataset\s*\("([^"]+)"\)\s*\{', data):
        name = m.group(1)
        block = data[m.start():_match_brace(data, m.end() - 1) + 1]
        if "(" in name:
            continue
        if fields is not None and name in fields:
            region = re.search(r'validity\s*=\s*\[\s*"([^"]+)"', block).group(1)
            vidx = mesh.region_vertex_indices.get(region)
            vm = re.search(r"Values\s*\((\d+)\)\s*\{(.*?)\}", block, re.DOTALL)
            if vidx is not None and len(vidx) == int(vm.group(1)):
                vals = np.asarray(fields[name], dtype=float)[vidx]
                body = " " + " ".join(f"{v:.15e}" for v in vals) + "\n    "
                block = block[:vm.start(2)] + body + block[vm.end(2):]
        keep.append(block)
        names.append(f'"{name}"')
        funcs.append(re.search(r"function\s*=\s*(\S+)", block).group(1))
    info = re.sub(r"datasets\s*=\s*\[.*?\]", "datasets    = [ " + " ".join(names) + " ]",
                  info, flags=re.DOTALL)
    info = re.sub(r"functions\s*=\s*\[.*?\]", "functions   = [ " + " ".join(funcs) + " ]",
                  info, flags=re.DOTALL)
    dst.write_text("DF-ISE text\n\nInfo {" + info + "}\n\nData {\n\n" + "\n\n".join(keep) + "\n\n}\n")


def _tdx(args: list[str], cwd: Path) -> None:
    from trifield.data.tdr_extract import find_tdx
    import subprocess
    tdx = find_tdx()
    if tdx is None:
        raise FileNotFoundError("Sentaurus 'tdx' not found")
    subprocess.run([str(tdx)] + args, cwd=cwd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Deck generation from the production sdevice template
# ---------------------------------------------------------------------------

def _replace_block(text: str, keyword: str, new_block: str) -> str:
    m = re.search(rf"^{keyword}\s*\{{", text, re.MULTILINE)
    if m is None:
        raise ValueError(f"no {keyword} block in template")
    end = _match_brace(text, m.end() - 1)
    return text[:m.start()] + new_block + text[end + 1:]


def production_base(params: SimulationParams) -> str:
    """The rendered production sdevice1 deck (physics, interfaces, math, plot)."""
    tmp = TEMPLATE_DIR / "sdevice1_des.cmd"
    mapping = params.mapping()
    mapping["AreaFactor"] = f"{area_factor(params.to_dict()):.10g}"
    text = tmp.read_text()
    for k, v in mapping.items():
        text = text.replace(f"@{k}@", str(v))
    return re.sub(r"@<(.*?)>@", lambda m: "0", text)       # ramp goals are replaced anyway


def make_deck(base: str, tag: str, vg: float, vd: float, solve: str,
              grid: str = "node_msh.tdr", math_extra: str = "") -> str:
    deck = _replace_block(base, "File", (
        "File{\n"
        f'\tGrid     = "{grid}"\n'
        '\tParameter= "parameter.par"\n'
        f'\tPlot     = "{tag}_des.tdr"\n'
        f'\tCurrent  = "{tag}_des.plt"\n'
        f'\tOutput   = "{tag}_des.log"\n'
        "}"))
    deck = _replace_block(deck, "Electrode", (
        "Electrode{\n"
        '\t{ Name="S" Voltage= 0.0 }\n'
        f'\t{{ Name="D" Voltage= {vd:.6g} }}\n'
        f'\t{{ Name="G" Voltage= {vg:.6g} WorkFunction= 5.2 }}\n'
        "}"))
    if math_extra:
        m = re.search(r"^Math\s*\{", deck, re.MULTILINE)
        deck = deck[:m.end()] + "\n\t" + math_extra + deck[m.end():]
    return _replace_block(deck, "Solve", "Solve{\n" + solve + "\n}")


EQUILIBRIUM = ("\tCoupled(Iterations= 1000){ Poisson }\n"
               "\tCoupled(Iterations= 1000){ Poisson Electron Hole }\n"
               "\tCoupled{ Poisson Electron Hole }\n")


def warm_solve(prefix: str, iterations: int = 40) -> str:
    """Newton straight from the predicted fields."""
    return f'\tLoad(FilePrefix= "{prefix}")\n\tCoupled(Iterations= {iterations}){{ Poisson Electron Hole }}'


def warm_poisson_first(prefix: str, iterations: int = 80) -> str:
    """Hold the predicted quasi-Fermi levels, make psi consistent with them (nonlinear
    Poisson, robust), then damped Newton on the full system."""
    return (f'\tLoad(FilePrefix= "{prefix}")\n'
            "\tCoupled(Iterations= 100){ Poisson }\n"
            f"\tCoupled(Iterations= {iterations} LineSearchDamping= 1e-4){{ Poisson Electron Hole }}")


def warm_ramp(prefix: str, vd: float) -> str:
    """Start from the prediction at the same gate bias and Vd = 0 (a much easier,
    near-equilibrium state) and ramp the drain only, with large adaptive steps."""
    return (f'\tLoad(FilePrefix= "{prefix}")\n'
            "\tCoupled(Iterations= 100){ Poisson }\n"
            "\tCoupled(Iterations= 80 LineSearchDamping= 1e-4){ Poisson Electron Hole }\n"
            "\tQuasistationary(\n\t\tInitialStep= 0.05 MinStep= 1e-6 MaxStep= 0.2\n"
            f'\t\tGoal{{ Name="D" Voltage= {vd:.6g} }}\n'
            "\t){ Coupled{ Poisson Electron Hole } }")


def neighbour_ramp(prefix: str, vg: float, vd: float) -> str:
    """Continue from a converged solution of this twin at a nearby bias: load it and ramp the
    electrodes to the target (the deck's electrode block carries the loaded point's bias)."""
    return (f'\tLoad(FilePrefix= "{prefix}")\n'
            "\tQuasistationary(\n\t\tInitialStep= 0.1 MinStep= 1e-6 MaxStep= 0.25\n"
            f'\t\tGoal{{ Name="G" Voltage= {vg:.6g} }} Goal{{ Name="D" Voltage= {vd:.6g} }}\n'
            "\t){ Coupled{ Poisson Electron Hole } }")


def cold_solve(vg: float, vd: float) -> str:
    """Equilibrium, then one adaptive ramp of both electrodes to the target bias.
    The electrode block of a cold deck starts at 0 V; the goals carry the bias."""
    return (EQUILIBRIUM +
            "\tQuasistationary(\n\t\tInitialStep= 0.01 MinStep= 1e-6 MaxStep= 0.1\n"
            f'\t\tGoal{{ Name="G" Voltage= {vg:.6g} }} Goal{{ Name="D" Voltage= {vd:.6g} }}\n'
            "\t){ Coupled{ Poisson Electron Hole } }")


# ---------------------------------------------------------------------------
# One design
# ---------------------------------------------------------------------------

class Twin:
    """Warm-start twin for one design (thread-safe point solves)."""

    def __init__(self, params, run_dir, models=None, par_file=None, mode: str = "warm",
                 device: str = "auto", max_parallel: int = 3, anchor: bool = True,
                 warm_recipes: tuple = ("direct",), warm_iterations: int = 40,
                 gate_uncertainty: float | None = None, refine: bool = True, refine_device: str = "cpu",
                 anchor_before_refine: bool = False, refine_config=None, feature_version: int | None = None,
                 cold_fallback: bool = True, refine_escalate=None, neighbour_ramp: bool = True,
                 metal_state: bool = True, vth_mode: str = "ladder3", refine_cpu_below: int = 10000):
        self.params = params if isinstance(params, SimulationParams) else SimulationParams.from_dict(params)
        self.run_dir = Path(run_dir)
        self.models = models
        self.mode = mode
        self.device = device
        self.par_file = par_file
        self.max_parallel = max_parallel
        self.anchor = anchor
        # physics refinement of the surrogate guess on Sentaurus' exact discrete equations
        # (trifield.physics): damped Gummel + joint Newton + hole solve, boundary rules from per-mesh exports
        self.refine = refine
        self.refine_device = refine_device
        self.anchor_before_refine = anchor_before_refine
        self.refine_config = refine_config
        # graph feature version the checkpoints were trained on (None = graph_builder default, v2;
        # pass 3 for checkpoints trained on v3 graphs)
        self.feature_version = feature_version
        # warm mode: fall back to the cold recipe when every warm recipe fails (can take hours on
        # large meshes; labelling runs disable it and drop the point instead)
        self.cold_fallback = cold_fallback
        # RefineConfig used when every warm recipe fails: refinement continues from the capped result with
        # this (larger) budget and the direct solve is retried once before any cold fallback
        self.refine_escalate = refine_escalate
        # continue a failed warm point from the twin's nearest converged solution before cold-starting
        self.neighbour_ramp = neighbour_ramp
        # write the contact-metal Fermi-potential drop into every guess (needs the physics export)
        self.metal_state = metal_state
        # Vth search: "legacy" brackets at 0/-2/-4/-6 V (the -2 V point is a deep off-state warm start: neighbour
        # ramps or cold solves of 300-1000 s); "ladder2" walks from 0 V in 0.5 V steps (down or up) to a tight
        # bracket near threshold, with the 0 V point solved alongside the on-state points; "ladder3" also starts
        # each regula-falsi point on small meshes from the nearest converged solution instead of surrogate +
        # refinement (there CPU refinement costs ~20 s, the extra Sentaurus Newton steps a few seconds)
        self.vth_mode = vth_mode
        # refine_device="auto": SuperLU on the CPU for small meshes (cuDSS is overhead-bound there: 14 s vs ~11-106 s
        # at 3k nodes), cuDSS on the GPU above `refine_cpu_below` vertices (17x faster at 25k nodes)
        self.refine_cpu_below = refine_cpu_below
        self._md = None
        self.solver_escalate = None
        self.solver = None
        self._refine_lock = threading.Lock()
        self.warm_recipes = tuple(warm_recipes)
        self.warm_iterations = warm_iterations
        self.gate_uncertainty = gate_uncertainty
        self.base = production_base(self.params)
        self.log: list[dict] = []          # one row per sdevice call
        self._lock = threading.Lock()
        self._n = 0
        self.mesh = self.mesh_fields = None
        self.template_dat = self.template_grd = None
        self.timing = {"mesh": 0.0, "prepare": 0.0, "predict": 0.0, "guess": 0.0, "physics_export": 0.0, "refine": 0.0}

    # ---- setup ---------------------------------------------------------------
    def mesh_design(self) -> None:
        """Run sde unless the run directory already holds node_msh.tdr."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if (self.run_dir / "node_msh.tdr").exists():
            return
        mapping = self.params.mapping()
        mapping["AreaFactor"] = f"{area_factor(self.params.to_dict()):.10g}"
        render_template(TEMPLATE_DIR / SDE_TEMPLATE, self.run_dir / SDE_TEMPLATE, mapping)
        t = run_timed(SDE_CMD + [SDE_TEMPLATE], self.run_dir, 3600,
                      self.run_dir / "sde.stdout.log", self.run_dir / "sde.stderr.log")
        self.timing["mesh"] = t["wall"]
        self.timing["mesh_cpu"] = t["cpu"]

    def prepare(self) -> None:
        """Parameter file, mesh parse, and a solution-structured template TDR."""
        from trifield.data.tdr_extract import convert_tdr_to_dfise, parse_dat, parse_grd, collect_node_fields
        t0 = time.monotonic()
        work = self.run_dir / "twin"
        work.mkdir(exist_ok=True)
        par = find_parameter_file(self.par_file) if self.par_file or not (self.run_dir / "parameter.par").exists() \
            else self.run_dir / "parameter.par"
        if par.resolve() != (work / "parameter.par").resolve():
            shutil.copy(par, work / "parameter.par")
        shutil.copy(self.run_dir / "node_msh.tdr", work / "node_msh.tdr")
        grd, dat = convert_tdr_to_dfise(work / "node_msh.tdr", work, "msh")
        self.mesh = parse_grd(grd)
        _mds = parse_dat(dat)
        self.mesh_fields = collect_node_fields(_mds, self.mesh)
        from trifield.data.tdr_extract import collect_region_fields
        self.mesh.region_fields = collect_region_fields(_mds, self.mesh)
        # Solve on the DF-ISE round-tripped mesh: a guess written through DF-ISE only
        # loads on a grid that went through the same conversion (Sentaurus compares
        # the geometry byte-for-byte). Validated: identical terminal currents to the
        # production mesh, cold and warm.
        _rewrite_dat(dat, work / "msh_clean.dat")
        _tdx(["--dfise2tdr", grd.name, "msh_clean.dat", GRID], work)
        if not (work / GRID).exists():
            raise RuntimeError("grid conversion failed")
        physics = None
        if self.mode == "warm" and (self.refine or self.metal_state):
            # the per-mesh exports are independent sdevice runs: overlap them with the template run
            physics = threading.Thread(target=self.prepare_physics, daemon=True)
            physics.start()
        if self.mode == "warm":
            # a zero-iteration run writes a TDR with the exact per-region dataset
            # layout sdevice expects to Load; its values are overwritten later
            deck = make_deck(self.base, "tmpl", 0.0, 0.0,
                             "\tCoupled(Iterations= 0){ Poisson Electron Hole }\n\tPlot(FilePrefix= \"tmpl\")",
                             grid=GRID, math_extra="RHSMin= 1e30")
            (work / "tmpl.cmd").write_text(deck)
            run_timed(SDEVICE + ["tmpl.cmd"], work, 600, work / "tmpl.out", work / "tmpl.err", check=False)
            tmpl = next(iter(sorted(work.glob("tmpl*_des.tdr"))), None)
            if tmpl is None:
                raise RuntimeError(f"template dump failed, see {work / 'tmpl.out'}")
            self.template_grd, self.template_dat = convert_tdr_to_dfise(tmpl, work, "tmpl")
        self.timing["prepare"] = time.monotonic() - t0
        if physics is not None:
            physics.join()

    def prepare_physics(self) -> None:
        """Per-mesh exports for the physics refinement (box data + equilibrium constants; cached in
        work/physics). On failure the twin keeps working without refinement."""
        t0 = time.monotonic()
        try:
            from trifield.physics import RefineConfig, Solver, export_mesh_data
            md = export_mesh_data(self.run_dir / "twin", self.base, grid=GRID)
            self._md = md
            if self.refine_device == "auto":
                import torch
                self.refine_device = ("cpu" if md.n < self.refine_cpu_below or not torch.cuda.is_available()
                                      else "cuda")
            self.solver = Solver(md, self.refine_device, self.refine_config or RefineConfig())
            if self.refine_escalate is not None:
                self.solver_escalate = Solver(md, self.refine_device, self.refine_escalate)
        except Exception as exc:                                          # noqa: broad-except
            log.warning("physics export failed (%s); continuing without refinement", exc)
            self.solver = None
        self.timing["physics_export"] = time.monotonic() - t0

    def refine_fields(self, fields: dict, vg: float, vd: float, solver=None) -> tuple[dict, dict]:
        """Surrogate fields -> refined fields on the exact discrete equations (+ diagnostics)."""
        solver = solver or self.solver
        guess = np.stack([np.asarray(fields[k], float) for k in FIELD_FUNCS], 1)
        t0 = time.monotonic()
        try:
            if solver.bk.gpu:
                with self._refine_lock:
                    res = solver.refine(guess, vg, vd)
            else:
                res = solver.refine(guess, vg, vd)
        except Exception as exc:                                          # noqa: broad-except
            log.warning("refinement failed at Vg=%.3f Vd=%.3f (%s); using the raw guess", vg, vd, exc)
            return fields, {"error": str(exc)}
        dt = time.monotonic() - t0
        with self._lock:
            self.timing["refine"] += dt
        cur = solver.terminal_currents(res.psi, res.phin, res.phip, area_factor(self.params.to_dict()))
        info = {"converged": res.converged, "gummel": res.gummel_iters, "newton": res.newton_iters,
                "merit": res.merit, "time": dt, "id_refined": cur["D"]}
        return res.fields(), info

    def with_metal_state(self, fields: dict, vg: float, vd: float) -> dict:
        """Add the contact-metal state Sentaurus keeps alongside psi / Phi_n / Phi_p: the Fermi-potential drop
        in the conducting contact metals, driven by the current the fields carry into each contact
        (`Solver.metal_drop`). Written to QuasiFermiPotential and applied to psi, Phi_n, Phi_p on metal
        vertices. Without it the metal equation starts ~1e9 off and deep off-state warm starts fail."""
        if self.solver is None or not self.metal_state:
            return fields
        md = self.solver.md
        f = {k: np.asarray(fields[k], float) for k in FIELD_FUNCS}
        try:
            drop = self.solver.metal_drop(f[FIELD_FUNCS[0]], f[FIELD_FUNCS[1]], f[FIELD_FUNCS[2]])
        except Exception as exc:                                          # noqa: broad-except
            log.warning("metal state failed at Vg=%.3f Vd=%.3f (%s)", vg, vd, exc)
            return fields
        out = dict(fields)
        target = md.electrode_bias(vg, vd) + drop                 # metal Fermi potential
        psi, phin, phip = (f[k].copy() for k in FIELD_FUNCS)
        # metal/semiconductor (ohmic) interface vertices: Phi_n = Phi_p = metal level; psi keeps its offset to
        # Phi (shift relative to the guess' own value: a refined guess pins them at V, a Sentaurus solution
        # already carries the drop)
        itf = md.metal_v & md.semi
        psi[itf] += target[itf] - phin[itf]
        phin[itf] = phip[itf] = target[itf]
        # metal-only vertices: psi = metal level + work-function offset; Sentaurus' Phi_n / Phi_p there are
        # placeholders (0) and stay as they are
        mo = md.metal_only
        off = np.where(md.metal_material == "TiN", md.gate_offset, md.gold_offset)
        psi[mo] = target[mo] + off[mo]
        out[FIELD_FUNCS[0]], out[FIELD_FUNCS[1]], out[FIELD_FUNCS[2]] = psi, phin, phip
        out["QuasiFermiPotential"] = electrode_fermi_levels(self.mesh, vg, vd) + drop
        return out

    # ---- surrogate -------------------------------------------------------------
    def predict(self, vg: float, vd: float, anchor: bool | None = None) -> dict:
        """Ensemble-mean fields at one bias point, keyed by Sentaurus dataset name."""
        import torch
        from trifield.data.graph_builder import build_graph
        from trifield.data.tdr_extract import ExtractedSimulation
        from trifield.inference import predict_ensemble
        t0 = time.monotonic()
        sim = ExtractedSimulation(mesh=self.mesh, mesh_fields=self.mesh_fields,
                                  solution_fields={}, sim_dir=self.run_dir)
        p = self.params.to_dict()
        kw = {} if self.feature_version is None else {"feature_version": self.feature_version}
        g = build_graph(sim, vg=vg, vd=vd, params=p, fin_pitch=float(p["Fw"]) + SFIN, **kw)
        gpu = next(self.models[0].parameters()).is_cuda
        if gpu:
            # one ensemble pass at a time per twin: concurrent passes on 50k+-node meshes ran out of
            # GPU memory when several twins share a device
            with self._refine_lock:
                mean, sigma = predict_ensemble(self.models, g)
                mean, sigma = mean.cpu(), sigma.cpu()
                torch.cuda.empty_cache()
        else:
            mean, sigma = predict_ensemble(self.models, g)
        from trifield.inference import ensemble_uncertainty
        self.last_uncertainty = ensemble_uncertainty(sigma, g) if len(self.models) > 1 else 0.0
        if self.anchor if anchor is None else anchor:
            mean = anchor_quasi_fermi(mean, g)
        self.timing["predict"] += time.monotonic() - t0
        return {FIELD_FUNCS[i]: mean[:, i].numpy() for i in range(3)}

    def write_guess(self, fields: dict, prefix: str, vg: float | None = None,
                    vd: float | None = None) -> None:
        """Prediction -> DF-ISE -> TDR on the round-tripped grid."""
        t0 = time.monotonic()
        work = self.run_dir / "twin"
        self._bias = getattr(self, "_bias", {})
        self._bias[prefix] = (vg if vg is not None else fields.get("_vg", 0.0),
                              vd if vd is not None else fields.get("_vd", 0.0))
        fields = {k: v for k, v in dict(fields).items() if not k.startswith("_")}
        if "QuasiFermiPotential" not in fields:
            fields["QuasiFermiPotential"] = electrode_fermi_levels(self.mesh, *self._bias[prefix])
        _rewrite_dat(self.template_dat, work / f"{prefix}.dat", self.mesh, fields)
        _tdx(["--dfise2tdr", self.template_grd.name, f"{prefix}.dat", f"{prefix}_des.tdr"], work)
        if not (work / f"{prefix}_des.tdr").exists():
            raise RuntimeError(f"writing the initial guess {prefix} failed")
        with self._lock:
            self.timing["guess"] += time.monotonic() - t0

    # ---- one bias point ----------------------------------------------------------
    def _sdevice(self, tag: str, deck: str) -> tuple[float | None, dict]:
        work = self.run_dir / "twin"
        (work / f"{tag}.cmd").write_text(deck)
        t = run_timed(SDEVICE + [f"{tag}.cmd"], work, 7200, work / f"{tag}.out", work / f"{tag}.err",
                      check=False)
        out = (work / f"{tag}.out").read_text(errors="replace")
        t["newton"] = len(re.findall(r"^\s*(\d+)\s+[-\d.]+e[+-]\d+", out, re.MULTILINE))
        cur = None
        if t["returncode"] == 0 and (work / f"{tag}_des.plt").exists():
            try:
                cur = abs(float(PLTFile(work / f"{tag}_des.plt").columns["d_total_current"][-1]))
                cur = cur if math.isfinite(cur) else None
            except Exception:                              # noqa: broad-except
                cur = None
        return cur, t

    def _neighbour(self, vg: float, vd: float):
        """(prefix, vg, vd) of the nearest converged solve (same Vd strongly preferred), or None."""
        with self._lock:
            done = [r for r in self.log if r["converged"] and r.get("plot")]
        if not done:
            return None
        r = min(done, key=lambda r: abs(r["vg"] - vg) + 10.0 * abs(r["vd"] - vd))
        return r["plot"], r["vg"], r["vd"]

    def solve_point(self, vg: float, vd: float, fields: dict | None = None,
                    refine: bool | None = None, fallbacks: bool = True) -> float | None:
        """Drain current [mA/mm] at (vg, vd); None if every recipe fails. ``refine`` overrides the
        twin-level refinement switch for this point.

        Warm mode tries, in order, until one converges (failed attempts still
        count towards the twin's runtime):
          1. direct Newton from the prediction at (vg, vd)
          2. Poisson with the predicted quasi-Fermi levels, then damped Newton
          3. prediction at (vg, 0) and a drain-only ramp to vd
          4. the cold recipe (equilibrium + one ramp of both electrodes)
        """
        with self._lock:
            self._n += 1
            tag = f"{self.mode}_{self._n:02d}"
        recipes = []
        unc = None
        if self.mode == "warm":
            refine = self.refine if refine is None else refine
            supplied = fields is not None
            if fields is None:
                fields = self.predict(vg, vd, anchor=self.anchor_before_refine if refine else None)
                unc = getattr(self, "last_uncertainty", None)
            refine_info = None
            if refine and self.solver is not None:
                fields, refine_info = self.refine_fields(fields, vg, vd)
            gated = (self.gate_uncertainty is not None and unc is not None and unc > self.gate_uncertainty)
            if not gated:
                # the metal drop follows the current the guess carries: trust it after a converged refinement
                # (exact currents) or for supplied fields (e.g. a Sentaurus solution); an unconverged refinement
                # can carry a current of the wrong sign, and pinning the metal at the bias is then the safer guess
                if (refine_info or {}).get("converged") or (supplied and refine_info is None):
                    fields = self.with_metal_state(fields, vg, vd)
                self.write_guess(fields, f"g{tag}", vg, vd)
                if "direct" in self.warm_recipes:
                    recipes.append(("direct", vg, vd, warm_solve(f"g{tag}", self.warm_iterations)))
                if "poisson_first" in self.warm_recipes:
                    recipes.append(("poisson_first", vg, vd, warm_poisson_first(f"g{tag}")))
                if "drain_ramp" in self.warm_recipes and vd > 0:
                    self.write_guess(self.predict(vg, 0.0), f"g{tag}v0", vg, 0.0)
                    recipes.append(("drain_ramp", vg, 0.0, warm_ramp(f"g{tag}v0", vd)))
        if fallbacks and self.mode == "warm" and self.neighbour_ramp and recipes and recipes[0][0] == "direct":
            # first fallback: the nearest converged solution itself as the guess (metal state recomputed for this
            # bias) - a plain warm start, much cheaper than a ramp or a cold solve when the surrogate guess fails
            recipes.insert(1, ("nbguess", vg, vd, None))
        nb = self._neighbour(vg, vd) if self.mode == "warm" and self.neighbour_ramp and fallbacks else None
        if nb is not None:
            recipes.append(("neighbour", nb[1], nb[2], neighbour_ramp(nb[0], vg, vd)))
        if fallbacks and (self.mode != "warm" or self.cold_fallback or not recipes):
            recipes.append(("cold", 0.0, 0.0, cold_solve(vg, vd)))
        if not fallbacks:
            recipes = [r for r in recipes if r[0] == "direct"] or recipes
        if (fallbacks and self.mode == "warm" and refine and self.solver_escalate is not None and refine_info
                and not refine_info.get("converged") and "error" not in refine_info):
            recipes.insert(len(recipes) - (recipes[-1][0] == "cold"), ("escalated", vg, vd, None))
        cur, total_wall, total_cpu, newton, used = None, 0.0, 0.0, 0, None
        for i, (name, ev_g, ev_d, solve) in enumerate(recipes):
            if name == "nbguess":
                nbf = self._neighbour_fields(vg, vd, max_dv=0.6)
                if nbf is None:
                    continue
                self.write_guess(self.with_metal_state(nbf, vg, vd), f"g{tag}n", vg, vd)
                solve = warm_solve(f"g{tag}n", self.warm_iterations)
                recipes[i] = (name, ev_g, ev_d, solve)
            if name == "escalated":
                try:
                    fields, info2 = self.refine_fields(fields, vg, vd, solver=self.solver_escalate)
                    refine_info = {**refine_info, "escalated": info2}
                    if info2.get("converged"):
                        fields = self.with_metal_state(fields, vg, vd)
                    self.write_guess(fields, f"g{tag}e", vg, vd)
                except Exception as exc:                                  # noqa: broad-except
                    log.warning("escalation failed at Vg=%.3f Vd=%.3f (%s)", vg, vd, exc)
                    continue
                solve = warm_solve(f"g{tag}e", self.warm_iterations)
                recipes[i] = (name, ev_g, ev_d, solve)
            deck = make_deck(self.base, f"{tag}{chr(97 + i)}", ev_g, ev_d, solve, grid=GRID)
            cur, t = self._sdevice(f"{tag}{chr(97 + i)}", deck)
            total_wall += t["wall"]
            total_cpu += t["cpu"]
            newton += t["newton"]
            if cur is not None:
                used = name
                break
        plot = f"{tag}{chr(97 + recipes.index(next(r for r in recipes if r[0] == used)))}" if used else None
        row = {"tag": tag, "vg": vg, "vd": vd, "current": cur, "converged": cur is not None, "uncertainty": unc,
               "plot": plot,
               "refine": refine_info if self.mode == "warm" else None,
               "recipe": used, "attempts": recipes.index(next(r for r in recipes if r[0] == used)) + 1
               if used else len(recipes), "newton": newton, "wall": total_wall, "cpu": total_cpu}
        with self._lock:
            self.log.append(row)
        return cur

    # ---- objectives ----------------------------------------------------------------
    def vth_search(self, vg11: float, i_on: float | None, tol_log: float = 0.01,
                   max_evals: int = 8) -> float | None:
        """Vg where Id(Vg, 12 V) = 1 mA/mm: bracket from Vg11 downwards, then Illinois
        regula falsi on log10(Id). Every evaluation is one solve."""
        f = lambda cur: math.log10(max(cur, 1e-30)) - math.log10(VTH_TARGET)
        pts = []
        if i_on is not None:
            pts.append((vg11, f(i_on)))
        if not pts or pts[-1][1] <= 0:
            return None if not pts else vg11            # never turns on above target
        lo = hi = None
        for vg in (0.0, -2.0, -4.0, -6.0):
            cur = self.solve_point(vg, VD_HIGH)
            if cur is None:
                continue
            pts.append((vg, f(cur)))
            if pts[-1][1] <= 0:
                lo = pts[-1]
                hi = min((p for p in pts if p[1] > 0), key=lambda p: p[0])
                break
        if lo is None:
            return None
        side = 0
        for _ in range(max_evals):
            (xa, fa), (xb, fb) = lo, hi
            x = xb - fb * (xb - xa) / (fb - fa)
            if abs(xb - xa) < 2e-3:
                break
            cur = self.solve_point(x, VD_HIGH)
            if cur is None:
                x = 0.5 * (xa + xb)
                cur = self.solve_point(x, VD_HIGH)
                if cur is None:
                    break
            fx = f(cur)
            if abs(fx) < tol_log:
                return x
            if fx > 0:
                hi = (x, fx)
                if side == +1:
                    lo = (lo[0], lo[1] / 2)
                side = +1
            else:
                lo = (x, fx)
                if side == -1:
                    hi = (hi[0], hi[1] / 2)
                side = -1
        (xa, fa), (xb, fb) = lo, hi
        return xb - fb * (xb - xa) / (fb - fa)

    def _neighbour_fields(self, vg: float, vd: float, max_dv: float = 0.3):
        """Fields of the nearest converged solve at the same Vd within `max_dv` (a Sentaurus solution: a better
        guess than surrogate + refinement for a small bias step, and no refinement cost)."""
        from trifield.data.tdr_extract import extract_solution
        with self._lock:
            done = [r for r in self.log if r["converged"] and r.get("plot") and r["vd"] == vd and abs(r["vg"] - vg) <= max_dv]
        if not done:
            return None
        r = min(done, key=lambda r: abs(r["vg"] - vg))
        tdr = self.run_dir / "twin" / f"{r['plot']}_des.tdr"
        if not tdr.exists():
            return None
        sol = extract_solution(tdr, self.mesh)
        return {k: np.asarray(sol[k], float) for k in FIELD_FUNCS}

    def vth_search_ladder2(self, vg11: float, i_on: float | None, tol_log: float = 0.01, max_evals: int = 8,
                           prefetched: dict | None = None, step: float = 0.5, from_neighbours: bool = False) -> float | None:
        """Vth on Sentaurus currents (same criterion and tolerance as `vth_search`). From the 0 V point step
        down (device on at 0 V) or up (off at 0 V) in `step` volts until the current crosses 1 mA/mm, so the
        bracket is tight and near threshold - where log Id is nearly linear in Vg and every warm start is easy -
        then Illinois regula falsi on log10(Id). Avoids both the deep off-state bracket point (-2 V: neighbour
        ramps or cold solves of 300-1000 s) and slow secant steps from the saturated on-state point."""
        f = lambda cur: math.log10(max(cur, 1e-30)) - math.log10(VTH_TARGET)
        if i_on is None:
            return None
        pts = {float(vg11): f(i_on)}
        if pts[vg11] <= 0:
            return vg11
        for vg, cur in (prefetched or {}).items():
            if cur is not None:
                pts[float(vg)] = f(cur)

        def at(vg, near=False):
            if vg not in pts:
                cur = None
                # neighbour-solution guesses pay off on small meshes (cheap Sentaurus Newton steps, costly CPU
                # refinement); on large meshes surrogate + GPU refinement needs fewer Sentaurus steps
                small = self._md is not None and self._md.n < self.refine_cpu_below
                if near and from_neighbours and small:
                    nbf = self._neighbour_fields(vg, VD_HIGH)
                    if nbf is not None:
                        cur = self.solve_point(vg, VD_HIGH, fields=nbf, refine=False, fallbacks=False)
                if cur is None:
                    cur = self.solve_point(vg, VD_HIGH)
                if cur is not None:
                    pts[vg] = f(cur)
            return pts.get(vg)

        f0 = at(0.0)
        if f0 is None:
            return self.vth_search(vg11, i_on)
        if f0 > 0:                                                       # on at 0 V: walk down
            ladder = [round(-step * k, 6) for k in range(1, 5)] + [-3.0, -4.0, -5.0, -6.0]
            for vg in ladder:
                fv = at(vg)
                if fv is not None and fv <= 0:
                    break
        else:                                                            # off at 0 V: walk up
            vg = step
            while vg < vg11:
                fv = at(round(vg, 6))
                if fv is not None and fv > 0:
                    break
                vg += step
        neg = [v for v, fv in pts.items() if fv <= 0]
        if not neg:
            return None
        lo = max(neg)
        pos = [v for v, fv in pts.items() if fv > 0 and v > lo]
        if not pos:
            return None
        hi = min(pos)
        if abs(pts[lo]) < tol_log:
            return lo
        if abs(pts[hi]) < tol_log:
            return hi
        (xa, fa), (xb, fb) = (lo, pts[lo]), (hi, pts[hi])
        side = 0
        for _ in range(max_evals):
            x = xb - fb * (xb - xa) / (fb - fa)
            if abs(xb - xa) < 2e-3:
                break
            fx = at(round(x, 6), near=True)
            if fx is None:
                x = 0.5 * (xa + xb)
                fx = at(round(x, 6), near=True)
                if fx is None:
                    break
            if abs(fx) < tol_log:
                return x
            if fx > 0:
                xb, fb = x, fx
                if side == +1:
                    fa /= 2
                side = +1
            else:
                xa, fa = x, fx
                if side == -1:
                    fb /= 2
                side = -1
        return xb - fb * (xb - xa) / (fb - fa)

    def run(self) -> dict:
        t0 = time.monotonic()
        self.mesh_design()
        self.prepare()
        vg11 = float(self.params.Vg11)
        prefetch = self.vth_mode != "legacy" and self.max_parallel >= 3 and vg11 > 0
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            f_on = pool.submit(self.solve_point, vg11, VD_HIGH)
            f_lin = pool.submit(self.solve_point, vg11, VD_LIN)
            f_zero = pool.submit(self.solve_point, 0.0, VD_HIGH) if prefetch else None
            i_on = f_on.result()
            i_lin = f_lin.result()
            pre = {0.0: f_zero.result()} if f_zero is not None else {}
        if self.vth_mode in ("ladder2", "ladder3"):
            vth = self.vth_search_ladder2(vg11, i_on, prefetched=pre, from_neighbours=self.vth_mode == "ladder3")
        else:
            vth = self.vth_search(vg11, i_on)
        wall = time.monotonic() - t0
        solves = self.log
        return {
            "mode": self.mode,
            "id_max": i_on,
            "ron": (VD_LIN / i_lin * 1e3) if i_lin else None,
            "vth": vth,
            "wall_total": wall,
            "wall_without_mesh": wall - self.timing.get("mesh", 0.0),
            "sdevice_wall_sum": sum(r["wall"] for r in solves),
            "sdevice_cpu_sum": sum(r["cpu"] for r in solves),
            "n_solves": len(solves),
            "n_failed": sum(not r["converged"] for r in solves),
            "recipes": ",".join(str(r.get("recipe")) for r in solves),
            "newton_sum": sum(r["newton"] for r in solves),
            **{f"t_{k}": v for k, v in self.timing.items()},
            "solves": solves,
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Warm-start digital twin for one design")
    ap.add_argument("--design", required=True, help="JSON design parameters")
    ap.add_argument("--ckpts", nargs="+", default=None, help="surrogate checkpoints (warm mode)")
    ap.add_argument("--par", default=None, help="Sentaurus parameter file")
    ap.add_argument("--run_dir", default="runs/twin_design", help="reused if it already has node_msh.tdr")
    ap.add_argument("--mode", choices=["warm", "cold"], default="warm")
    ap.add_argument("--device", default="auto", help="surrogate device")
    ap.add_argument("--refine_device", default="auto", help="physics refinement device; auto = CPU (SuperLU) below "
                    "10k vertices, GPU (cuDSS, ~17x faster at 25k nodes) above")
    ap.add_argument("--refine_cap", type=int, default=30, help="refinement Newton budget before Sentaurus "
                    "(0 = full RefineConfig); escalation continues to the full budget if Sentaurus fails")
    ap.add_argument("--no_escalate", action="store_true")
    ap.add_argument("--no_refine", action="store_true", help="raw surrogate guess (ablation)")
    ap.add_argument("--feature_version", type=int, default=None, help="graph features of the checkpoints")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    params = SimulationParams.from_dict(json.loads(Path(args.design).read_text()))
    models = None
    if args.mode == "warm":
        from trifield.inference import load_ensemble
        models = load_ensemble(args.ckpts, args.device)
    from trifield.physics import RefineConfig
    cfg = RefineConfig()
    if args.refine_cap:
        cfg.newton_maxit, cfg.restarts = args.refine_cap, 1
    res = Twin(params, args.run_dir, models, args.par, args.mode, args.device, refine=not args.no_refine,
               refine_device=args.refine_device, refine_config=cfg, feature_version=args.feature_version,
               refine_escalate=None if args.no_escalate else RefineConfig()).run()
    for r in res.pop("solves"):
        log.info("  solve %-8s Vg=%6.3f Vd=%5.2f  conv=%s newton=%3d  %.1f s", r["tag"], r["vg"], r["vd"],
                 r["converged"], r["newton"], r["wall"])
    log.info(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))


if __name__ == "__main__":
    main()
