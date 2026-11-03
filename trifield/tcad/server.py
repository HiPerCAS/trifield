"""Persistent Sentaurus Device process that serves many warm-start solves of one design.

Every ``sdevice`` invocation pays a fixed cost before its first Newton step: the driver script, binary start and
licence checkout (~0.9 s), and the device initialisation (grid, parameter file, box-method data: ~1.2 s at 3k
vertices, ~5.6 s at 13k), plus the File-section plot at exit. On enhancement-mode designs a warm-started point
converges in 5-7 Newton steps, so this fixed part was 60-70% of the twin's Sentaurus time.

Sentaurus' Tcl mode (``sdevice --tcl``, sdevice user guide ch. 6 "Tcl Command File") initialises a device once
(``sdevice_init``) and then runs any number of Solve sections (``sdevice_solve``) in the same process. The server
runs a Tcl loop that reads one request per line from stdin, runs its Solve section with the output redirected to
the request's own ``.out`` file and answers ``@@DONE`` on stdout. A request is a warm start at (Vg, Vd):

    NewCurrentPrefix= "<tag>_"                    # the request's own current file <tag>_des.plt
    Load(FilePrefix= "<guess>")                    # psi, Phi_n, Phi_p, metal Fermi potential
    Quasistationary( -DoZero -Extrapolate InitialStep= 1 MaxStep= 1 MinStep= 1  Goal{G} Goal{D} )
        { Coupled(Iterations= k){ Poisson Electron Hole } }
    Plot(FilePrefix= "<tag>")                      # only if the solve converged

Why each piece:

* ``Load`` restores the solution variables but not the bias (the guesses are written through DF-ISE, which carries
  no contact state), so the bias is set by a one-step quasistationary: ``-DoZero`` skips the t = 0 solve (it would
  solve at the previous request's bias), ``MinStep= 1`` makes a failed Newton fail the request instead of
  bisecting the step, and ``-Extrapolate`` keeps the loaded guess as the initial iterate. The Newton iterations,
  convergence test and iteration limit are those of the per-point deck's ``Coupled``.
* ``Math{ AutoCNPMinStepFactor= 0 AutoNPMinStepFactor= 0 }``: with MinStep = 1 every step is "smaller than
  2 MinStep", which by default switches on CNormPrint and a NewtonPlot TDR after every Newton step (0.5 s each at
  3k vertices). Both only produce diagnostics.
* ``sdevice_solve`` returns 1 even when the quasistationary failed, so success is decided by the current file: a
  converged step writes its row, a failed one writes none.

Validated against one process per point (8 designs, every twin point): identical terminal currents (bitwise),
identical Newton iteration counts and residual histories, identical plot TDR values; a failed request leaves no
trace in the next one. The plot of a freshly initialised device has exactly the dataset layout of the twin's
template, so the server also writes the template.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("trifield.tcad.server")

_TCL = r'''fconfigure stdout -buffering line
if {[catch {sdevice_init {
%(deck)s
}} msg]} {
    puts "@@FAIL init $msg"
    exit 1
}
puts "@@READY"
while {[gets stdin line] >= 0} {
    set id [lindex $line 0]
    if {$id eq "QUIT"} break
    set t0 [clock milliseconds]
    set rc [catch {
        set f [open [lindex $line 1] r]; set cmd [read $f]; close $f
        sdevice_solve $cmd >& [lindex $line 2]
    } ok]
    set plot [lindex $line 4]
    if {$rc == 0 && $plot ne "-" && [file exists [lindex $line 3]]} {
        catch {sdevice_solve "Solve{ Plot(FilePrefix= \"$plot\") }" >>& [lindex $line 2]}
    }
    puts "@@DONE $id $rc [expr {[clock milliseconds]-$t0}]"
}
catch {sdevice_finish}
puts "@@BYE"
'''


def server_deck(deck: str, current: str = "des.plt", output: str = "server_des.log") -> str:
    """A complete sdevice deck (``twin.make_deck`` output) -> the ``sdevice_init`` part of a server: no Solve
    section, no File-section plot (the final state is not needed), every request's current file
    ``<prefix>_des.plt`` (NewCurrentPrefix + ``current``), and the automatic Newton diagnostics off."""
    from trifield.tcad.twin import _replace_block
    deck = _replace_block(deck, "Solve", "")
    deck = re.sub(r"^\s*Plot\s*=.*$\n?", "", deck, count=1, flags=re.MULTILINE)
    deck = re.sub(r'^(\s*Current\s*=\s*)"[^"]*"', rf'\1"{current}"', deck, count=1, flags=re.MULTILINE)
    deck = re.sub(r'^(\s*Output\s*=\s*)"[^"]*"', rf'\1"{output}"', deck, count=1, flags=re.MULTILINE)
    m = re.search(r"^Math\s*\{", deck, re.MULTILINE)
    return deck[:m.end()] + "\n\tAutoCNPMinStepFactor= 0\n\tAutoNPMinStepFactor= 0" + deck[m.end():]


def warm_request(prefix: str, vg: float, vd: float, iterations: int = 40) -> str:
    """Solve section of one warm start served by `SdeviceServer.solve` (the server adds NewCurrentPrefix)."""
    return (f'\tLoad(FilePrefix= "{prefix}")\n'
            "\tQuasistationary( -DoZero -Extrapolate InitialStep= 1 MaxStep= 1 MinStep= 1\n"
            f'\t\tGoal{{ Name="G" Voltage= {vg:.6g} }} Goal{{ Name="D" Voltage= {vd:.6g} }}\n'
            f"\t){{ Coupled(Iterations= {iterations}){{ Poisson Electron Hole }} }}")


_MARK = re.compile(r"@@(READY|DONE|FAIL|BYE)\b")


def _cpu_seconds(pid: int) -> float:
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
    except Exception:                                                    # noqa: broad-except
        return 0.0


class SdeviceServer:
    """One persistent ``sdevice --tcl`` process in ``work`` (thread-safe: one request at a time).

    ``start()`` launches it; ``solve()`` runs one Solve section and returns (current file, info) or None when the
    process is gone (the caller then runs the point as a separate sdevice process). Requests write
    ``<tag>.cmd`` (the Solve section), ``<tag>.out`` (its output), ``<tag>_des.plt`` and, when converged and
    ``plot`` is set, ``<tag>_des.tdr`` - the same files as a per-point deck named <tag>."""

    def __init__(self, work, init_deck: str, name: str = "srv0", threads: int = 1, cmd: tuple = ("sdevice",)):
        self.work = Path(work)
        self.name = name
        self.threads = int(threads)
        self.cmd = list(cmd)
        self.init_deck = init_deck
        self.proc = None
        self.dead = False
        self._ready = threading.Event()
        self._lock = threading.Lock()                 # one request at a time
        self._cv = threading.Condition()
        self._done: dict[str, list] = {}
        self._n = 0
        self.t_start = None
        self.t_ready = None
        self.n_requests = 0
        self.cpu_requests = 0.0

    # ---- process ---------------------------------------------------------------
    def start(self) -> "SdeviceServer":
        (self.work / f"{self.name}.tcl").write_text(_TCL % {"deck": self.init_deck})
        self.t_start = time.monotonic()
        self.proc = subprocess.Popen(
            self.cmd + ["-n", "--threads", str(self.threads), "--tcl", f"{self.name}.tcl"], cwd=self.work,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(self.work / f"{self.name}.err", "wb"),
            text=True, bufsize=1)
        threading.Thread(target=self._reader, daemon=True, name=f"{self.name}-reader").start()
        return self

    def _reader(self) -> None:
        out = open(self.work / f"{self.name}.out", "w")
        try:
            for line in self.proc.stdout:
                m = _MARK.search(line)                   # sdevice's own (unflushed) output may precede the marker
                if m is None:
                    out.write(line)
                    continue
                out.write(line[:m.start()])
                f = line[m.start():].split()
                if f[0] == "@@READY":
                    self.t_ready = time.monotonic()
                    self._ready.set()
                elif f[0] == "@@DONE" and len(f) >= 4:
                    with self._cv:
                        self._done[f[1]] = f[2:]
                        self._cv.notify_all()
                elif f[0] == "@@FAIL":
                    out.write(line)
        except Exception:                                                # noqa: broad-except
            pass
        finally:
            out.close()
            self.dead = True
            self._ready.set()
            with self._cv:
                self._cv.notify_all()

    @property
    def alive(self) -> bool:
        return self.proc is not None and not self.dead and self.proc.poll() is None

    def wait_ready(self, timeout: float = 600.0) -> bool:
        self._ready.wait(timeout)
        return self.alive and self._ready.is_set()

    def cpu(self) -> float:
        """CPU seconds of the process so far (all threads)."""
        return _cpu_seconds(self.proc.pid) if self.proc is not None else 0.0

    def close(self, timeout: float = 60.0) -> float:
        """Stop the process; returns its total CPU seconds (measured before it exits)."""
        if self.proc is None:
            return 0.0
        cpu = self.cpu()
        if self.proc.poll() is None and self._lock.acquire(timeout=timeout):
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout)
            except Exception:                                            # noqa: broad-except
                pass
            finally:
                self._lock.release()
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.dead = True
        return cpu

    # ---- requests ----------------------------------------------------------------
    def solve(self, tag: str, solve: str, plot: bool = True, timeout: float = 7200.0):
        """Run one Solve section; None when the process is (or goes) down before answering.
        Returns (plt path, {"wall", "cpu", "ms"}); the request converged iff the plt has a row."""
        with self._lock:
            if not self.wait_ready() or not self.alive:
                return None
            self._n += 1
            rid = f"r{self._n}"
            plt = self.work / f"{tag}_des.plt"
            for p in (plt, self.work / f"{tag}_des.tdr"):
                if p.exists():
                    p.unlink()
            (self.work / f"{tag}.cmd").write_text(f'Solve{{\n\tNewCurrentPrefix= "{tag}_"\n{solve}\n}}\n')
            c0, t0 = self.cpu(), time.monotonic()
            try:
                self.proc.stdin.write(f"{rid} {tag}.cmd {tag}.out {plt.name} {tag if plot else '-'}\n")
                self.proc.stdin.flush()
            except Exception:                                            # noqa: broad-except
                self.dead = True
                return None
            with self._cv:
                while rid not in self._done and not self.dead and time.monotonic() - t0 < timeout:
                    self._cv.wait(1.0)
                res = self._done.pop(rid, None)
            if res is None:                                              # died or timed out
                log.warning("sdevice server %s: no answer to %s (%s)", self.name, tag,
                            "timeout" if not self.dead else "process gone")
                if self.proc.poll() is None:
                    self.proc.kill()
                self.dead = True
                return None
            wall, cpu = time.monotonic() - t0, self.cpu() - c0
            self.n_requests += 1
            self.cpu_requests += cpu
            return plt, {"wall": wall, "cpu": cpu, "ms": float(res[1]), "tcl_rc": int(res[0])}


class ServerPool:
    """Up to n `SdeviceServer`s of one design; `acquire()` hands out an idle live server (waiting while all live
    ones are busy) or None when none is alive."""

    def __init__(self, servers):
        self.servers = list(servers)
        self._free = list(self.servers)
        self._cv = threading.Condition()

    def acquire(self, timeout: float = 7200.0):
        t0 = time.monotonic()
        with self._cv:
            while True:
                live = [s for s in self.servers if s.alive or (not s.dead and not s._ready.is_set())]
                if not live:
                    return None
                free = [s for s in self._free if s in live]
                if free:
                    s = free[0]
                    self._free.remove(s)
                    return s
                if time.monotonic() - t0 > timeout:
                    return None
                self._cv.wait(1.0)

    def release(self, server) -> None:
        with self._cv:
            if server not in self._free:
                self._free.append(server)
            self._cv.notify_all()

    def close(self) -> float:
        """Stop every server; total CPU seconds of all of them."""
        return sum(s.close() for s in self.servers)
