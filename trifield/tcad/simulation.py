"""Run one Sentaurus simulation (or only its mesh) for a design and build graphs.

A design is rendered into the command templates under ``templates/`` by
``@name@`` substitution, then

    sde       builds the multi-fin structure and the tetrahedral mesh (node_msh.tdr)
    sdevice   #1: Id-Vd corner sweep, writing nine TDR field snapshots + node1_des.plt
    sdevice   #2: backward Vg sweep for the threshold voltage (node2_des.plt)

``Simulation.run`` returns the simulator's I_d,max and V_th and can convert the
snapshots into training graphs. ``MeshOnlySimulation`` runs ``sde`` only and
yields the nine bias graphs with zero targets, which is all the surrogate
needs to score a candidate (~30 s instead of a 20-60 min solve).

    python -m trifield.tcad.simulation --design examples/design.json --par parameter.par \
        --work_dir runs --out_dir data/mine [--mesh_only] [--keep]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import resource
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from trifield.params import FIXED, area_factor, fin_pitch, snapshot_biases, snapshot_stem

log = logging.getLogger("trifield.tcad")

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
SDE_TEMPLATE = "sde_dvs.cmd"
SDEVICE_TEMPLATES = ("sdevice1_des.cmd", "sdevice2_des.cmd")
SDE_CMD = ["sde", "-e", "-nowait", "-noSyntaxCheck", "-l"]
SDEVICE_CMD = ["sdevice", "-q", "-n", "--exit-on-failure", "--threads", "1"]
MESH_TDR = "node_msh.tdr"


class SimulationError(RuntimeError):
    """A Sentaurus tool returned a non-zero exit code."""


# ---------------------------------------------------------------------------
# Design parameters
# ---------------------------------------------------------------------------

@dataclass
class SimulationParams:
    GaN: float
    AlGaN: float
    Fw: float
    Tox: float
    Hg: float
    AlGaN_doping: float
    ch_doping: float = FIXED["ch_doping"]
    Vg11: float = FIXED["Vg11"]
    Lext2: float = FIXED["Lext2"]
    Nfins: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationParams":
        names = {f.name for f in fields(cls)}
        kw = {k: (int(v) if k == "Nfins" else float(v)) for k, v in d.items() if k in names}
        return cls(**kw)

    def to_dict(self) -> dict:
        return asdict(self)

    def mapping(self) -> dict[str, str]:
        """``@name@`` substitution values for the command templates."""
        return {k: (str(v) if isinstance(v, int) else f"{v:.12g}") for k, v in asdict(self).items()}


def _eval_expr(expr: str, mapping: dict) -> str:
    for k, v in mapping.items():
        expr = re.sub(rf"\b{k}\b", str(v), expr)
    return str(eval(expr, {"__builtins__": {}}, {}))    # arithmetic on numbers only


def render_template(src: Path, dst: Path, mapping: dict) -> None:
    """Substitute ``@name@`` tokens and evaluate ``@<expr>@`` arithmetic."""
    text = Path(src).read_text()
    for k, v in mapping.items():
        text = text.replace(f"@{k}@", str(v))
    text = re.sub(r"@<(.*?)>@", lambda m: _eval_expr(m.group(1), mapping), text)
    Path(dst).write_text(text)


def find_parameter_file(par) -> Path:
    if par is not None:
        p = Path(par)
        if p.exists():
            return p
        raise FileNotFoundError(f"parameter file not found: {p}")
    p = TEMPLATE_DIR / "parameter.par"
    if p.exists():
        return p
    raise FileNotFoundError(
        "No Sentaurus material parameter file. The GaN/AlN/AlGaN parameter sets are part "
        "of the Synopsys MaterialDB and cannot be redistributed: assemble parameter.par "
        "from your installation as described in trifield/tcad/README.md and pass --par.")


# ---------------------------------------------------------------------------
# .plt (DF-ISE current file) parsing
# ---------------------------------------------------------------------------

class PLTFile:
    """Columns of a Sentaurus ``.plt`` current file as ``{name: np.ndarray}``."""

    def __init__(self, path):
        raw = Path(path).read_text().replace("\n", "")
        keys = re.findall(r'"([a-zA-Z0-9 .]+)"', raw)
        values = raw.split("Data {")[1].split("}")[0].split()
        keys = [self._snake(k) for k in keys]
        cols: dict[str, list[float]] = {k: [] for k in keys}
        for i, v in enumerate(values):
            cols[keys[i % len(keys)]].append(float(v))
        self.columns = {k: np.asarray(v) for k, v in cols.items()}

    @staticmethod
    def _snake(s: str) -> str:
        return "".join("_" + c.lower() if c.isupper() else c for c in s.replace(" ", "")).lstrip("_")

    def max(self, col: str) -> float:
        return float(self.columns[col].max())

    def find_vth(self, vcol: str = "g_outer_voltage", ycol: str = "d_total_current",
                 target: float = 1.0) -> float:
        """Gate voltage at which ``ycol`` crosses ``target`` (log-linear interpolation,
        linear extrapolation outside the sweep)."""
        y, v = self.columns[ycol], self.columns[vcol]
        good = y > 0
        ly, lv = np.log10(y[good]), v[good]
        order = np.argsort(ly)
        ly, lv = ly[order], lv[order]
        t = np.log10(target)
        if t < ly[0]:
            i0, i1 = 0, 1
        elif t > ly[-1]:
            i0, i1 = len(ly) - 2, len(ly) - 1
        else:
            return float(np.interp(t, ly, lv))
        slope = (lv[i1] - lv[i0]) / (ly[i1] - ly[i0])
        return float(lv[i0] + slope * (t - ly[i0]))


def ron_from_plt(plt: "PLTFile", vd: float = 3.0) -> float | None:
    """On-resistance [Ohm mm] = Vd / Id at the first logged point with drain bias ~ vd
    (output-curve ramp of sdevice1 at Vg = Vg11)."""
    vdc = plt.columns.get("d_outer_voltage")
    idc = plt.columns.get("d_total_current")
    if vdc is None or idc is None:
        return None
    i = int(np.argmin(np.abs(vdc - vd)))
    if abs(vdc[i] - vd) > 1e-3 or abs(idc[i]) < 1e-30:
        return None
    return float(vd / abs(idc[i])) * 1e3       # V / (mA/mm) -> Ohm mm


# ---------------------------------------------------------------------------
# Graph construction from TDR snapshots
# ---------------------------------------------------------------------------

def build_snapshot_graphs(sim_dir, params, out_dir, design_name: str | None = None,
                          mesh_tdr: str = MESH_TDR, require_solution: bool = True) -> list[Path]:
    """Convert the bias-snapshot TDRs of a finished simulation into ``<design>__<snap>.pt``.

    With ``require_solution=False`` missing solution TDRs give mesh-only graphs
    (zero targets) for every bias point, which is the candidate-pool case.
    """
    import torch
    from trifield.data.graph_builder import build_graph
    from trifield.data.tdr_extract import ExtractedSimulation, extract_mesh, extract_solution

    sim_dir, out_dir = Path(sim_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params.to_dict() if isinstance(params, SimulationParams) else dict(params)
    design_name = design_name or sim_dir.name
    mesh, mesh_fields = extract_mesh(sim_dir, mesh_tdr=mesh_tdr)
    written = []
    for tag, vg, vd in snapshot_biases(params):
        stem = snapshot_stem(tag)
        tdr = sim_dir / f"{stem}.tdr"
        if tdr.exists():
            solution = extract_solution(tdr, mesh)
        elif require_solution:
            log.warning("%s: snapshot %s missing, skipped", design_name, tdr.name)
            continue
        else:
            solution = {}
        sim = ExtractedSimulation(mesh=mesh, mesh_fields=mesh_fields,
                                 solution_fields=solution, sim_dir=sim_dir)
        data = build_graph(sim, vg=vg, vd=vd, params=params, fin_pitch=fin_pitch(params))
        data.sim_name, data.snapshot_name = design_name, stem
        out = out_dir / f"{design_name}__{stem}.pt"
        torch.save(data, out)
        written.append(out)
    log.info("%s: wrote %d graphs to %s", design_name, len(written), out_dir)
    return written


# ---------------------------------------------------------------------------
# Simulation drivers
# ---------------------------------------------------------------------------

def run_timed(cmd, cwd, timeout, stdout_path, stderr_path, check: bool = True) -> dict:
    """Run a tool, return its wall-clock and CPU (user + sys of the child) seconds."""
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
        try:
            proc = subprocess.run(cmd, cwd=cwd, check=check, timeout=timeout, stdout=out, stderr=err)
        except subprocess.CalledProcessError as e:
            raise SimulationError(f"{Path(cmd[0]).name} exited with {e.returncode} in {cwd}") from None
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"wall": time.monotonic() - t0,
            "cpu": (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime),
            "returncode": proc.returncode}


class _Run:
    def __init__(self, params, par_file=None, work_dir="runs", timeout_sec: int = 2400,
                 keep: bool = False, name: str | None = None):
        self.params = params if isinstance(params, SimulationParams) else SimulationParams.from_dict(params)
        self.par_file = find_parameter_file(par_file)
        self.timeout_sec = timeout_sec
        self.keep = keep
        self.name = name or f"sim_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        self.sim_dir = (Path(work_dir) / self.name).resolve()
        if self.sim_dir.exists():                 # stale partial run: start clean
            shutil.rmtree(self.sim_dir)
        self._t0 = None
        self.timings: dict[str, dict] = {}      # template -> {"wall": s, "cpu": s}

    def _prepare(self) -> dict:
        self.sim_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.par_file, self.sim_dir / "parameter.par")
        self._t0 = time.monotonic()
        mapping = self.params.mapping()
        mapping["AreaFactor"] = f"{area_factor(self.params.to_dict()):.10g}"
        return mapping

    def _run_tool(self, cmd_base: list[str], template: str, mapping: dict) -> None:
        dst = self.sim_dir / template
        render_template(TEMPLATE_DIR / template, dst, mapping)
        remaining = self.timeout_sec - (time.monotonic() - self._t0)
        if remaining <= 0:
            raise TimeoutError(f"{self.name}: global timeout exceeded before {template}")
        cmd = cmd_base + [str(dst)]
        log.info("%s: %s", self.name, " ".join(cmd))
        self.timings[template] = run_timed(cmd, self.sim_dir, remaining,
                                           self.sim_dir / f"{template}.stdout.log",
                                           self.sim_dir / f"{template}.stderr.log")

    def cleanup(self) -> None:
        if not self.keep and self.sim_dir.exists():
            shutil.rmtree(self.sim_dir, ignore_errors=True)


class Simulation(_Run):
    """Full pipeline: mesh + two sdevice runs; returns the simulator's FoMs."""

    def run(self, graphs_dir=None, design_name: str | None = None) -> dict:
        try:
            mapping = self._prepare()
            self._run_tool(SDE_CMD, SDE_TEMPLATE, mapping)
            for tpl in SDEVICE_TEMPLATES:
                self._run_tool(SDEVICE_CMD, tpl, mapping)
            id_max = PLTFile(self.sim_dir / "node1_des.plt").max("d_total_current")
            vth = PLTFile(self.sim_dir / "node2_des.plt").find_vth()
            ron = None
            try:
                plt1 = PLTFile(self.sim_dir / "node1_des.plt")
                ron = ron_from_plt(plt1)
            except Exception:                         # noqa: broad-except
                pass
            result = {"name": self.name, "id_max": id_max, "vth": vth, "ron": ron,
                      "wall_sec": round(time.monotonic() - self._t0, 1),
                      "timings": self.timings, "graphs": []}
            if graphs_dir is not None:
                result["graphs"] = [str(p) for p in build_snapshot_graphs(
                    self.sim_dir, self.params, graphs_dir, design_name or self.name)]
            return result
        finally:
            self.cleanup()


class MeshOnlySimulation(_Run):
    """``sde`` only: the nine bias graphs of a design with zero targets (candidate pool)."""

    def run(self, graphs_dir, design_name: str | None = None) -> list[Path]:
        try:
            mapping = self._prepare()
            self._run_tool(SDE_CMD, SDE_TEMPLATE, mapping)
            return build_snapshot_graphs(self.sim_dir, self.params, graphs_dir,
                                         design_name or self.name, require_solution=False)
        finally:
            self.cleanup()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Sentaurus for one design and build its graphs")
    ap.add_argument("--design", required=True, help="JSON file with the design parameters")
    ap.add_argument("--par", default=None, help="Sentaurus material parameter file (see tcad/README.md)")
    ap.add_argument("--work_dir", default="runs")
    ap.add_argument("--out_dir", default="data/mine", help="where the <design>__<snapshot>.pt graphs go")
    ap.add_argument("--name", default=None, help="design name (default: sim_<timestamp>)")
    ap.add_argument("--mesh_only", action="store_true", help="sde only: candidate graphs with zero targets")
    ap.add_argument("--keep", action="store_true", help="keep the Sentaurus working directory")
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    params = SimulationParams.from_dict(json.loads(Path(args.design).read_text()))
    log.info("design: %s", params)
    if args.mesh_only:
        files = MeshOnlySimulation(params, args.par, args.work_dir, args.timeout, args.keep,
                                   args.name).run(args.out_dir)
        log.info("mesh-only graphs: %s", [f.name for f in files])
    else:
        res = Simulation(params, args.par, args.work_dir, args.timeout, args.keep,
                         args.name).run(args.out_dir)
        log.info("TCAD: Id,max = %.1f mA/mm, Vth = %.3f V (%.0f s)", res["id_max"], res["vth"], res["wall_sec"])
        with (Path(args.out_dir) / "tcad_foms.csv").open("a", newline="") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["name", "id_max", "vth", "wall_sec"] + [k.name for k in fields(SimulationParams)])
            w.writerow([res["name"], res["id_max"], res["vth"], res["wall_sec"]]
                       + list(params.to_dict().values()))


if __name__ == "__main__":
    main()
