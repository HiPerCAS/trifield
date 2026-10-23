#!/usr/bin/env python
"""Generate ``assets/framework.gif``, the animated overview of TriField.

Run from the repository root (the paths below are relative to it):

    python assets/make_framework_gif.py                 # -> assets/framework.gif
    python assets/make_framework_gif.py --out /tmp/f.gif --device cpu
    python assets/make_framework_gif.py --models "models/pignn_ens_seed*.pt" \
                                        --models_dataonly "models/dataonly_seed*.pt"

Scenes: title; design parameters -> Sentaurus mesh graph; message passing
(1..8-hop receptive field); prediction vs TCAD for the physics-informed and the
data-only ensemble with |error| maps and per-field RMSE; bias sweep over the
nine snapshots (data-only ensemble); ensemble disagreement (physics-informed
ensemble, the one used in the active-learning loop); size transfer of both
ensembles to the 5-fin device; the TCAD-in-the-loop active-learning loop;
closing frame.

Only the bundled example graphs (``examples/data``) and checkpoints
(``models``) are needed; rendering uses the matplotlib ``Agg`` backend and
the GIF is assembled with Pillow.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from trifield import load_ensemble, load_graph, predict_ensemble  # noqa: E402
from trifield.data.graph_builder import MATERIALS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DESIGN_1FIN = "sim_1775923714283611604_ec8890a7"
DESIGN_5FIN = "sim_1775948900113442451_cf94f0fe"
SNAPSHOTS = ["snap_vg02_vd00_des", "snap_vg04_vd00_des", "snap_vg06_vd00_des",
             "snap_vg08_vd00_des", "snap_vg10_vd00_des", "snap_vg10_vd03_des",
             "snap_vg10_vd06_des", "snap_vg10_vd09_des", "node1_des"]
DEFAULT_MODELS = [str(ROOT / "models" / "pignn_ens_seed0.pt"), str(ROOT / "models" / "pignn_ens_seed1.pt")]
DEFAULT_MODELS_DATAONLY = [str(ROOT / "models" / "dataonly_seed0.pt"), str(ROOT / "models" / "dataonly_seed1.pt")]
DEFAULT_DATA = str(ROOT / "examples" / "data")
DEFAULT_OUT = str(ROOT / "assets" / "framework.gif")

# ----------------------------------------------------------------------------- style
W, H, DPI = 900, 520, 100
INK = "#1c2b3a"          # headings / body text
MUTED = "#6b7785"        # secondary text, axis labels
ACCENT = "#e4701e"       # highlight colour
ACCENT_SOFT = "#fbe3d0"
PI_COLOR = "#2b6cb0"     # physics-informed ensemble (titles / numbers)
DO_COLOR = "#e4701e"     # data-only ensemble
BOX_FILL = "#f3f5f8"
BOX_EDGE = "#aab4c0"
EDGE_GREY = "#c4c9cf"
MAT_COLORS = {"GaN": "#3b74b9", "AlGaN": "#f0963c", "Gold": "#d8b12e",
              "HfO2": "#5aa469", "TiN": "#7d7d7d"}
PSI_CMAP, ERR_CMAP, SIG_CMAP = "viridis", "magma", "magma"
PSI, PHIN, PHIP, SIGV = r"$\psi$", r"$\Phi_n$", r"$\Phi_p$", r"$\sigma_v$"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#8a949e",
    "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelcolor": MUTED,
    "axes.labelsize": 8.5, "axes.titlesize": 10, "axes.titlecolor": INK,
    "mathtext.fontset": "dejavusans",
})


# ----------------------------------------------------------------------------- mesh layers
class Layer:
    """One mesh layer (all nodes on a coordinate plane) as a 2-D triangulation.

    The meshes are layered: the 1-fin example has 12 distinct y values and the
    fin interior has no nodes, so a thin slab through the fin would be empty.
    We therefore take a full plane of nodes and draw its real connectivity:
    edges are ``edge_index`` restricted to the layer, triangles are the
    3-cliques of those edges (= the mesh faces lying in the plane). Stored
    coordinates: x = layer stack (growth direction), y = fin width / fin
    stacking, z = channel (source at z = 0, drain at z = -1 um).
    """

    def __init__(self, g, axis: int, value: float | None, horiz: int, vert: int,
                 flip_h: bool = True):
        pos = g.pos.numpy()
        if value is None:  # densest layer along ``axis``
            vals, counts = np.unique(np.round(pos[:, axis], 4), return_counts=True)
            value = float(vals[np.argmax(counts)])
        self.value = value
        self.idx = np.where(np.abs(pos[:, axis] - value) < 1e-3)[0]
        self.h = (-1.0 if flip_h else 1.0) * pos[self.idx, horiz]
        self.v = pos[self.idx, vert]
        self.mat = g.vertex_material.numpy()[self.idx]
        n_all = g.num_nodes
        local = np.full(n_all, -1, dtype=np.int64)
        local[self.idx] = np.arange(len(self.idx))
        ei = g.edge_index.numpy()
        a, b = local[ei[0]], local[ei[1]]
        keep = (a >= 0) & (b >= 0) & (a < b)
        a, b = a[keep], b[keep]
        self.edges = np.stack([a, b], 1)
        nbrs = [set() for _ in range(len(self.idx))]
        for i, j in zip(a.tolist(), b.tolist()):
            nbrs[i].add(j)
            nbrs[j].add(i)
        tris = [(i, j, k) for i, j in zip(a.tolist(), b.tolist())
                for k in nbrs[i] & nbrs[j] if k > j]
        self.tri = mtri.Triangulation(self.h, self.v, triangles=np.asarray(tris, dtype=np.int64))
        self.segments = np.stack([np.stack([self.h[a], self.v[a]], 1),
                                  np.stack([self.h[b], self.v[b]], 1)], 1)
        pad_h, pad_v = 0.02 * np.ptp(self.h), 0.06 * np.ptp(self.v)
        self.xlim = (self.h.min() - pad_h, self.h.max() + pad_h)
        self.ylim = (self.v.min() - pad_v, self.v.max() + pad_v)

    def values(self, full: torch.Tensor | np.ndarray) -> np.ndarray:
        arr = full.numpy() if isinstance(full, torch.Tensor) else np.asarray(full)
        return arr[self.idx]


def hop_distances(g, seed: int, kmax: int) -> np.ndarray:
    """BFS hop distance from ``seed`` on the full 3-D graph (-1 beyond ``kmax``)."""
    ei = g.edge_index.numpy()
    dist = np.full(g.num_nodes, -1, dtype=np.int64)
    dist[seed] = 0
    frontier = np.array([seed])
    for k in range(1, kmax + 1):
        cand = np.unique(ei[1][np.isin(ei[0], frontier)])
        new = cand[dist[cand] < 0]
        if len(new) == 0:
            break
        dist[new] = k
        frontier = new
    return dist


# ----------------------------------------------------------------------------- drawing helpers
def new_fig():
    return plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")


def header(fig, title: str, step: str | None = None):
    fig.text(0.04, 0.937, title, fontsize=15.5, fontweight="bold", color=INK, va="center")
    fig.add_artist(Line2D([0.04, 0.96], [0.892, 0.892], color=ACCENT, lw=1.3,
                          transform=fig.transFigure))
    if step:
        fig.text(0.96, 0.937, step, fontsize=9, color=MUTED, ha="right", va="center")


def caption(fig, text: str, y: float = 0.055, size: float = 11):
    fig.text(0.5, y, text, fontsize=size, color=INK, ha="center", va="center")


def render(fig) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    img = Image.fromarray(buf)
    if img.size != (W, H):
        img = img.resize((W, H), Image.Resampling.LANCZOS)
    return img


def style_axes(ax, layer: Layer, xlabel: str | None, ylabel: str | None, aspect="equal"):
    ax.set_aspect(aspect)
    ax.set_xlim(*layer.xlim)
    ax.set_ylim(*layer.ylim)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, pad=2)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=1)


def field_map(ax, layer: Layer, vals: np.ndarray, vmin: float, vmax: float, cmap: str):
    return ax.tripcolor(layer.tri, vals, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax,
                        rasterized=True)


def colorbar(fig, rect, im, label: str, ticks=None):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, ticks=ticks)
    cb.outline.set_edgecolor("#8a949e")
    cb.ax.tick_params(labelsize=8, length=2.5, pad=2)
    cb.set_label(label, fontsize=8.5, color=MUTED, labelpad=2)
    return cb


def draw_graph(ax, layer: Layer, node_mask=None, edge_lw: float = 0.35, node_size: float = 5.0):
    """Mesh-graph drawing: thin grey edges and nodes coloured by material."""
    colors = np.array([MAT_COLORS[m] for m in MATERIALS] + ["#000000"])
    if node_mask is None:
        node_mask = np.ones(len(layer.idx), dtype=bool)
    emask = node_mask[layer.edges[:, 0]] & node_mask[layer.edges[:, 1]]
    if emask.any():
        ax.add_collection(LineCollection(layer.segments[emask], colors=EDGE_GREY,
                                         linewidths=edge_lw, zorder=1))
    ax.scatter(layer.h[node_mask], layer.v[node_mask], c=colors[layer.mat[node_mask]],
               s=node_size, linewidths=0, zorder=3)


def material_legend(ax, loc="upper right", **kw):
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=MAT_COLORS[m], label=m)
               for m in MATERIALS]
    leg = ax.legend(handles=handles, loc=loc, fontsize=8, frameon=True, framealpha=0.9,
                    edgecolor="#d0d5db", handletextpad=0.3, borderpad=0.5, **kw)
    leg.get_frame().set_linewidth(0.6)
    return leg


def fmt_param(name: str, v: float) -> str:
    if name == "AlGaN_doping":
        mant, exp = f"{v:.3g}".split("e")
        return f"{mant}e{int(exp)} cm$^{{-3}}$"
    return f"{v:.3g} um"


def rmse_all(mean: torch.Tensor, g) -> np.ndarray:
    """Per-field RMSE [V] over all nodes of the snapshot."""
    return (mean - g.y).pow(2).mean(0).sqrt().numpy()


def short_tag(paths) -> str:
    """'a_seed0.pt, a_seed1.pt' -> 'a_seed{0,1}.pt' when the names differ in one character."""
    names = [Path(p).name for p in paths]
    if len(names) == 2 and len(names[0]) == len(names[1]):
        diff = [i for i, (c0, c1) in enumerate(zip(*names)) if c0 != c1]
        if len(diff) == 1:
            i = diff[0]
            return f"{names[0][:i]}{{{names[0][i]},{names[1][i]}}}{names[0][i + 1:]}"
    return ", ".join(names)


def rmse_line(vals) -> str:
    return "   ".join(f"{sym} {v:.3f}" for sym, v in zip((PSI, PHIN, PHIP), vals))


# ----------------------------------------------------------------------------- scenes
def scene_title():
    frames = []
    for alpha in (0.45, 1.0):
        fig = new_fig()
        fig.text(0.5, 0.60, "TriField", fontsize=46, fontweight="bold", color=INK,
                 ha="center", va="center", alpha=alpha)
        fig.text(0.5, 0.455, "mesh-native physics-informed graph surrogate for TCAD",
                 fontsize=16, color=INK, ha="center", va="center", alpha=alpha)
        fig.add_artist(Line2D([0.30, 0.70], [0.395, 0.395], color=ACCENT, lw=2,
                              transform=fig.transFigure, alpha=alpha))
        fig.text(0.5, 0.325, "drift-diffusion fields at every mesh node  "
                 "•  deep-ensemble uncertainty  •  TCAD-in-the-loop design exploration",
                 fontsize=10.5, color=MUTED, ha="center", va="center", alpha=alpha)
        fig.text(0.5, 0.12, "ICCAD 2026  •  github.com/HiPerCAS/trifield", fontsize=9.5,
                 color=MUTED, ha="center", va="center", alpha=alpha)
        frames.append((render(fig), 180 if alpha < 1 else 950))
    return frames


def scene_design_to_graph(g, layer: Layer):
    p = g.params
    rows = [("T_GaN", "GaN", "GaN channel-layer thickness"),
            ("T_AlGaN", "AlGaN", "AlGaN barrier thickness"),
            ("W_fin", "Fw", "fin width"),
            ("T_ox", "Tox", "HfO2 gate-oxide thickness"),
            ("T_gate", "Hg", "TiN gate-metal thickness"),
            ("N_AlGaN", "AlGaN_doping", "barrier doping")]

    def base(step):
        fig = new_fig()
        header(fig, "Design  →  Sentaurus mesh  →  graph", step)
        fig.text(0.04, 0.835, f"Held-out 1-fin design ({int(g.nfins)} fin, "
                 f"{g.num_nodes:,} nodes, {g.num_edges // 2:,} edges)",
                 fontsize=10.5, color=INK, fontweight="bold", va="center")
        for c, start in ((0.04, 0), (0.52, 3)):
            for r, (sym, key, desc) in enumerate(rows[start:start + 3]):
                y = 0.775 - 0.052 * r
                fig.text(c, y, sym, fontsize=9.5, color=INK, va="center", family="DejaVu Sans Mono")
                fig.text(c + 0.095, y, fmt_param(key, float(p[key])), fontsize=9.5, color=ACCENT,
                         va="center", fontweight="bold")
                fig.text(c + 0.235, y, desc, fontsize=8.5, color=MUTED, va="center")
        return fig

    frames = []
    # (a) parameters only
    fig = base("design")
    fig.text(0.5, 0.40, "Sentaurus SDE meshes the device (tetrahedra, ~30 s) ...", fontsize=11,
             color=MUTED, ha="center", va="center", style="italic")
    frames.append((render(fig), 800))

    # (b) progressive reveal of the mesh graph along the channel, then the full graph
    rect = [0.065, 0.165, 0.90, 0.44]
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        fig = base("mesh graph")
        ax = fig.add_axes(rect)
        mask = layer.h <= layer.h.min() + t * np.ptp(layer.h) + 1e-6
        draw_graph(ax, layer, node_mask=mask)
        style_axes(ax, layer, "source → drain [um]", "layer stack [um]")
        ax.set_title(f"mesh graph, cross-section through the fin (y = {layer.value:.2f} um layer, "
                     f"{len(layer.idx)} nodes, {len(layer.edges)} edges)", fontsize=9.5, pad=4)
        material_legend(ax, ncol=5, loc="upper right")
        if t >= 1.0:
            caption(fig, "Sentaurus mesh = graph: nodes = vertices (17 features), edges = tet edges")
            frames.append((render(fig), 900))
        else:
            frames.append((render(fig), 100))
    return frames


def scene_message_passing(g, layer: Layer):
    # seed: a GaN (channel) node of the layer under the gate centre
    target = np.array([0.25, 0.065])
    gan = np.where(layer.mat == MATERIALS.index("GaN"))[0]
    seed_local = gan[np.argmin(np.hypot(layer.h[gan] - target[0], layer.v[gan] - target[1]))]
    seed = int(layer.idx[seed_local])
    kmax = 8
    dist_full = hop_distances(g, seed, kmax)
    dist = dist_full[layer.idx]
    zoom = (0.09, 0.41)
    cmap = plt.get_cmap("YlOrRd_r")
    frames = []
    for k in range(1, kmax + 1):
        fig = new_fig()
        header(fig, "Message passing: 8 edge-conditioned GATv2 blocks", f"block {k} of {kmax}")
        # overview (left, full device) with the zoom window
        ax0 = fig.add_axes([0.05, 0.60, 0.52, 0.25])
        draw_graph(ax0, layer, edge_lw=0.25, node_size=2.5)
        style_axes(ax0, layer, None, None)
        ax0.tick_params(labelbottom=False, labelleft=False, length=0)
        ax0.add_patch(Rectangle((zoom[0], layer.ylim[0]), zoom[1] - zoom[0],
                                layer.ylim[1] - layer.ylim[0], fill=False, ec=ACCENT, lw=1.2))
        ax0.set_title("cross-section (zoom window in orange)", fontsize=9, pad=3)
        # zoom (right): nodes within k hops of the seed
        ax = fig.add_axes([0.60, 0.15, 0.37, 0.70])
        reached = (dist >= 0) & (dist <= k)
        ax.add_collection(LineCollection(layer.segments, colors=EDGE_GREY, linewidths=0.4, zorder=1))
        emask = reached[layer.edges[:, 0]] & reached[layer.edges[:, 1]]
        if emask.any():
            ax.add_collection(LineCollection(layer.segments[emask], colors=ACCENT, linewidths=0.9,
                                             zorder=2, alpha=0.8))
        ax.scatter(layer.h[~reached], layer.v[~reached], c="#d5d9de", s=7, linewidths=0, zorder=3)
        cols = cmap(dist[reached] / kmax)
        ax.scatter(layer.h[reached], layer.v[reached], c=cols, s=22, linewidths=0.4,
                   edgecolors=INK, zorder=4)
        ax.scatter([layer.h[seed_local]], [layer.v[seed_local]], marker="*", s=160, c=ACCENT,
                   edgecolors=INK, linewidths=0.6, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(*zoom)
        ax.set_ylim(*layer.ylim)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2.5, pad=2)
        ax.set_xlabel("source → drain [um]", labelpad=1)
        ax.set_title(f"receptive field of the ★ node after {k} block{'s' if k > 1 else ''}: "
                     f"{int((dist_full >= 0).sum() if k == kmax else ((dist_full >= 0) & (dist_full <= k)).sum())} "
                     f"nodes (3-D)", fontsize=9, pad=3)
        # explanation (left, below the overview)
        lines = ["Each block updates a node from its mesh neighbours:",
                 "attention weights conditioned on the edge vector",
                 "(distance, displacement, material change).",
                 f"After {k} block{'s' if k > 1 else ''} a node sees its {k}-hop neighbourhood;",
                 "no global pooling, so the same weights run on any mesh."]
        for i, s in enumerate(lines):
            fig.text(0.05, 0.50 - 0.052 * i, s, fontsize=9.5, color=INK if i in (0, 3) else MUTED,
                     va="center")
        fig.text(0.05, 0.16, "hop:", fontsize=8.5, color=MUTED, va="center")
        for j in range(kmax + 1):
            fig.add_artist(Line2D([0.10 + 0.038 * j], [0.16], marker="o", ms=7, ls="",
                                  color=cmap(j / kmax), mec=INK, mew=0.4, transform=fig.transFigure))
            fig.text(0.10 + 0.038 * j, 0.115, str(j), fontsize=7.5, color=MUTED, ha="center")
        caption(fig, "encoder MLP  →  8 × GATv2 (4 heads, residual + LayerNorm)  →  "
                     "decoder MLP  →  (" + PSI + ", " + PHIN + ", " + PHIP + ") per node", y=0.055, size=10)
        frames.append((render(fig), 150 if k < kmax else 420))
    return frames


def scene_pred_vs_tcad(g, layer: Layer, mean_pi: torch.Tensor, mean_do: torch.Tensor,
                       rmse_pi: np.ndarray, rmse_do: np.ndarray, tag_pi: str, tag_do: str):
    """Row 1: physics-informed / data-only / TCAD psi (shared scale); row 2: both |error| maps
    on one scale plus the per-field RMSE table. Cross-sections drawn with 2x vertical scale."""
    truth = layer.values(g.y[:, 0])
    p_pi, p_do = layer.values(mean_pi[:, 0]), layer.values(mean_do[:, 0])
    e_pi, e_do = np.abs(p_pi - truth), np.abs(p_do - truth)
    vmin, vmax = float(g.y[:, 0].min()), float(g.y[:, 0].max())
    emax = float(np.percentile(np.concatenate([e_pi, e_do]), 99.5))
    cols, w, vex = (0.065, 0.355, 0.645), 0.255, 2.0
    row1 = [("physics-informed ensemble " + PSI + " [V]", p_pi, PI_COLOR),
            ("data-only ensemble " + PSI + " [V]", p_do, DO_COLOR),
            ("TCAD (Sentaurus Device) " + PSI + " [V]", truth, INK)]
    row2 = [("|error| physics-informed [V]", e_pi, PI_COLOR), ("|error| data-only [V]", e_do, DO_COLOR)]

    frames = []
    for stage in (0, 1, 2):   # 0: field maps, 1: + |error| maps, 2: + RMSE table and caption
        fig = new_fig()
        header(fig, "Prediction vs. TCAD on a held-out design", "Vg = %.0f V, Vd = %.0f V" % (g.vg, g.vd))
        im = None
        for i, (x0, (title, vals, col)) in enumerate(zip(cols, row1)):
            ax = fig.add_axes([x0, 0.555, w, 0.30])
            im = field_map(ax, layer, vals, vmin, vmax, PSI_CMAP)
            style_axes(ax, layer, None, "stack [um] (2×)" if i == 0 else None, aspect=vex)
            ax.tick_params(labelbottom=False, labelleft=(i == 0))
            ax.set_title(title, fontsize=9, pad=3, color=col, fontweight="bold")
        colorbar(fig, [0.905, 0.575, 0.011, 0.26], im, PSI + " [V]")
        if stage >= 1:
            for i, (x0, (title, vals, col)) in enumerate(zip(cols, row2)):
                ax = fig.add_axes([x0, 0.20, w, 0.30])
                im = field_map(ax, layer, vals, 0.0, emax, ERR_CMAP)
                style_axes(ax, layer, "source → drain [um]", "stack [um] (2×)" if i == 0 else None,
                           aspect=vex)
                ax.tick_params(labelleft=(i == 0))
                ax.set_title(title, fontsize=9, pad=3, color=col, fontweight="bold")
            colorbar(fig, [0.618, 0.22, 0.011, 0.26], im, "|error| [V]")
        else:
            fig.text(0.5, 0.33, f"all {g.num_nodes:,} nodes in a single forward pass of each ensemble",
                     fontsize=10.5, color=MUTED, ha="center", va="center", style="italic")
        if stage == 2:
            x0, xs = 0.68, (0.79, 0.895)
            fig.text(x0, 0.465, f"RMSE, all {g.num_nodes:,} nodes [V]", fontsize=9.5, color=INK,
                     fontweight="bold", va="center")
            fig.text(xs[0], 0.405, "physics-\ninformed", fontsize=8.5, color=PI_COLOR, ha="center",
                     va="center", fontweight="bold")
            fig.text(xs[1], 0.405, "data-\nonly", fontsize=8.5, color=DO_COLOR, ha="center",
                     va="center", fontweight="bold")
            for r, (sym, a, b) in enumerate(zip((PSI, PHIN, PHIP), rmse_pi, rmse_do)):
                y = 0.335 - 0.055 * r
                fig.text(x0 + 0.01, y, sym, fontsize=11, color=INK, va="center")
                fig.text(xs[0], y, f"{a:.3f}", fontsize=11, color=PI_COLOR, ha="center", va="center",
                         fontweight="bold")
                fig.text(xs[1], y, f"{b:.3f}", fontsize=11, color=DO_COLOR, ha="center", va="center",
                         fontweight="bold")
            fig.text(x0, 0.165, f"{tag_pi}  |  {tag_do}", fontsize=7.5, color=MUTED, va="center")
            caption(fig, "data-only: tightest fields inside the training range;  physics-informed: "
                         "current conservation and size transfer (paper, Fig. 3)", y=0.05, size=10)
            frames.append((render(fig), 1400))
        else:
            frames.append((render(fig), 550 if stage == 0 else 450))
    return frames


def scene_bias_sweep(snaps, layer: Layer, means, vmin: float, vmax: float, tag: str):
    frames = []
    pts = [(float(s.vg), float(s.vd)) for s in snaps]
    for i, (g, mean) in enumerate(zip(snaps, means)):
        fig = new_fig()
        header(fig, "Bias sweep: one model, nine bias points of the same mesh", f"{i + 1} / {len(snaps)}")
        fig.text(0.87, 0.125, "data-only ensemble\n" + tag, fontsize=8.5, color=DO_COLOR,
                 ha="center", va="center")
        ax = fig.add_axes([0.04, 0.555, 0.60, 0.30])
        im = field_map(ax, layer, layer.values(mean[:, 0]), vmin, vmax, PSI_CMAP)
        style_axes(ax, layer, None, "stack [um]")
        ax.tick_params(labelbottom=False)
        ax.set_title("predicted " + PSI + " [V]", fontsize=9.5, pad=3)
        ax2 = fig.add_axes([0.04, 0.195, 0.60, 0.30])
        field_map(ax2, layer, layer.values(g.y[:, 0]), vmin, vmax, PSI_CMAP)
        style_axes(ax2, layer, "source → drain [um]", "stack [um]")
        ax2.set_title("TCAD " + PSI + " [V]", fontsize=9.5, pad=3)
        colorbar(fig, [0.655, 0.195, 0.012, 0.66], im, PSI + " [V] (fixed scale)")
        # bias-point diagram
        axb = fig.add_axes([0.775, 0.35, 0.19, 0.42])
        axb.plot([p[1] for p in pts], [p[0] for p in pts], "o", ms=6, color="#c9d0d8", mec="#8a949e")
        axb.plot(pts[i][1], pts[i][0], "o", ms=11, color=ACCENT, mec=INK)
        axb.set_xlim(-1, 13)
        axb.set_ylim(0, 6)
        axb.set_xticks([0, 3, 6, 9, 12])
        axb.set_yticks([1, 2, 3, 4, 5])
        axb.set_xlabel("Vd [V]", labelpad=1)
        axb.set_ylabel("Vg [V]", labelpad=1)
        for s in ("top", "right"):
            axb.spines[s].set_visible(False)
        axb.set_title("bias point", fontsize=9.5, pad=3)
        fig.text(0.87, 0.235, f"Vg = {g.vg:.0f} V\nVd = {g.vd:.0f} V", fontsize=13, color=ACCENT,
                 fontweight="bold", ha="center", va="center")
        caption(fig, "Vg and Vd enter as node features: the gate turns the channel on, "
                     "Vd pulls the drain-side potential up", y=0.055, size=10.5)
        frames.append((render(fig), 170 if i < len(snaps) - 1 else 400))
    return frames


def scene_uncertainty(g, layer: Layer, sigma: torch.Tensor, mean: torch.Tensor, tag: str):
    sig = layer.values(sigma)
    err = np.abs(layer.values(mean[:, 0]) - layer.values(g.y[:, 0]))
    frames = []
    for with_error in (False, True):
        fig = new_fig()
        header(fig, "Uncertainty from ensemble disagreement", "physics-informed ensemble")
        fig.text(0.04, 0.105, f"physics-informed ensemble ({tag}): the ensemble used in the "
                 "active-learning loop", fontsize=8.5, color=PI_COLOR, va="center")
        ax = fig.add_axes([0.04, 0.555, 0.80, 0.30])
        im = field_map(ax, layer, sig, 0.0, float(np.percentile(sig, 99.5)), SIG_CMAP)
        style_axes(ax, layer, None if with_error else "source → drain [um]", "stack [um]")
        if with_error:
            ax.tick_params(labelbottom=False)
        ax.set_title("per-node ensemble disagreement " + SIGV + " [V]   (Vg = %.0f V, Vd = %.0f V)"
                     % (g.vg, g.vd), fontsize=9.5, pad=3)
        colorbar(fig, [0.855, 0.555, 0.012, 0.30], im, SIGV + " [V]")
        if with_error:
            ax2 = fig.add_axes([0.04, 0.195, 0.80, 0.30])
            im2 = field_map(ax2, layer, err, 0.0, float(np.percentile(err, 99.5)), ERR_CMAP)
            style_axes(ax2, layer, "source → drain [um]", "stack [um]")
            ax2.set_title("|error| vs TCAD [V] for comparison: high " + SIGV + " flags the unreliable regions",
                          fontsize=9.5, pad=3)
            colorbar(fig, [0.855, 0.195, 0.012, 0.30], im2, "|error| [V]")
        caption(fig, "deep-ensemble disagreement = per-node uncertainty  "
                     "(" + SIGV + "$^2$ = mean$_j$ ||f$_j$(v) − f̄(v)||$^2$ over the members)",
                y=0.055, size=10.5)
        frames.append((render(fig), 900 if with_error else 500))
    return frames


def scene_size_transfer(g5, layer5: Layer, mean5_pi: torch.Tensor, mean5_do: torch.Tensor,
                        rmse5_pi: np.ndarray, rmse5_do: np.ndarray, g1, tag_pi: str, tag_do: str):
    """Three top-view panels (physics-informed, data-only, TCAD) on a shared scale."""
    truth = layer5.values(g5.y[:, 0])
    p_pi, p_do = layer5.values(mean5_pi[:, 0]), layer5.values(mean5_do[:, 0])
    vmin, vmax = float(g5.y[:, 0].min()), float(g5.y[:, 0].max())
    cols, w = (0.035, 0.325, 0.615), 0.265
    panels = [("physics-informed ensemble " + PSI + " [V]", p_pi, PI_COLOR, rmse5_pi, tag_pi),
              ("data-only ensemble " + PSI + " [V]", p_do, DO_COLOR, rmse5_do, tag_do),
              ("TCAD (Sentaurus Device) " + PSI + " [V]", truth, INK, None, None)]
    frames = []
    for with_pred in (False, True):
        fig = new_fig()
        header(fig, "Mesh-native: the same weights on a 5-fin device", "size transfer")
        im = None
        for i, (x0, (title, vals, col, rm, tag)) in enumerate(zip(cols, panels)):
            ax = fig.add_axes([x0, 0.30, w, 0.56])
            ax.set_title(title, fontsize=9, pad=3, color=col, fontweight="bold")
            style_axes(ax, layer5, "source → drain [um]" if i == 1 else None,
                       "fin stacking direction [um]" if i == 0 else None)
            ax.tick_params(labelleft=(i == 0))
            if rm is not None and not with_pred:
                ax.text(0.5, 0.5, "ensemble\nforward pass\n(no retraining)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10.5, color=MUTED, style="italic")
                continue
            im = field_map(ax, layer5, vals, vmin, vmax, PSI_CMAP)
            xc = x0 + w / 2
            if rm is not None:
                fig.text(xc, 0.205, f"RMSE, all {g5.num_nodes:,} nodes [V]:", fontsize=8.5, color=MUTED,
                         ha="center", va="center")
                fig.text(xc, 0.16, rmse_line(rm), fontsize=9.5, color=col, ha="center", va="center",
                         fontweight="bold")
                fig.text(xc, 0.125, tag, fontsize=7.5, color=MUTED, ha="center", va="center")
            else:
                fig.text(xc, 0.205, "trained on 1–5-fin devices", fontsize=8.5, color=MUTED,
                         ha="center", va="center")
                fig.text(xc, 0.16, f"1 fin {g1.num_nodes:,} → 5 fins {g5.num_nodes:,} nodes",
                         fontsize=9, color=INK, ha="center", va="center", fontweight="bold")
                fig.text(xc, 0.125, "top view, y–z layer at x = %.2f um" % layer5.value, fontsize=7.5,
                         color=MUTED, ha="center", va="center")
        colorbar(fig, [0.905, 0.33, 0.011, 0.50], im, PSI + " [V]")
        caption(fig, "same weights, trained on 1–5 fins: 1 fin → 5 fins (12k nodes), no retraining",
                y=0.075, size=10.5)
        caption(fig, "beyond 5 fins the physics term keeps the derived device metrics usable (paper, Fig. 3)",
                y=0.035, size=9.5)
        frames.append((render(fig), 1300 if with_pred else 450))
    return frames


def scene_active_learning():
    boxes = [("Candidate pool", "mesh only, ~30 s each", (0.15, 0.74)),
             ("Ensemble inference", "seconds per design", (0.50, 0.74)),
             ("Acquisition score", "uncertainty " + SIGV + " + predicted objective", (0.85, 0.74)),
             ("Top-K to Sentaurus Device", "full solve, 20–60 min each", (0.67, 0.24)),
             ("Retrain the ensemble", "with the new TCAD solutions", (0.31, 0.24))]
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    bw, bh = 0.27, 0.26
    frames = []
    for active in list(range(len(boxes))) + [None]:
        fig = new_fig()
        header(fig, "TCAD-in-the-loop active learning", "design-space exploration")
        ax = fig.add_axes([0.03, 0.13, 0.94, 0.74])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for i, (name, detail, (cx, cy)) in enumerate(boxes):
            on = active is None or i == active
            ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                                        boxstyle="round,pad=0.012,rounding_size=0.03",
                                        fc=ACCENT_SOFT if (i == active) else BOX_FILL,
                                        ec=ACCENT if (i == active) else BOX_EDGE,
                                        lw=2.2 if (i == active) else 1.0, zorder=2))
            ax.text(cx, cy + 0.045, name, ha="center", va="center", fontsize=11,
                    fontweight="bold", color=INK if on else MUTED, zorder=3)
            ax.text(cx, cy - 0.05, detail, ha="center", va="center", fontsize=9,
                    color=INK if on else MUTED, zorder=3)
        for a, b in arrows:
            (ax0, ay0), (bx0, by0) = boxes[a][2], boxes[b][2]
            if abs(ay0 - by0) < 1e-6:  # horizontal
                sgn = 1 if bx0 > ax0 else -1
                p0, p1 = (ax0 + sgn * bw / 2 + 0.012 * sgn, ay0), (bx0 - sgn * bw / 2 - 0.012 * sgn, by0)
                conn = "arc3,rad=0"
            elif by0 < ay0:  # down (right column)
                p0, p1 = (ax0, ay0 - bh / 2 - 0.012), (bx0 + bw / 2 - 0.08, by0 + bh / 2 + 0.012)
                conn = "arc3,rad=0.25"
            else:  # up (left column)
                p0, p1 = (ax0 - bw / 2 + 0.08, ay0 + bh / 2 + 0.012), (bx0, by0 - bh / 2 - 0.012)
                conn = "arc3,rad=0.25"
            hot = active is not None and b == active
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                         lw=2.4 if hot else 1.4, color=ACCENT if hot else BOX_EDGE,
                                         connectionstyle=conn, zorder=1))
        ax.text(0.50, 0.46, "loop until the budget of TCAD solves is spent", ha="center",
                va="center", fontsize=9.5, color=MUTED, style="italic")
        caption(fig, "only the most uncertain / most promising designs are simulated; "
                     "the surrogate scores the whole pool", y=0.055, size=10.5)
        frames.append((render(fig), 230 if active is not None else 400))
    return frames


def scene_closing(speedup: str):
    frames = []
    for alpha in (0.5, 1.0):
        fig = new_fig()
        fig.text(0.5, 0.70, "TriField", fontsize=40, fontweight="bold", color=INK, ha="center",
                 va="center", alpha=alpha)
        fig.text(0.5, 0.555, speedup, fontsize=16, color=ACCENT, ha="center", va="center",
                 fontweight="bold", alpha=alpha)
        fig.add_artist(Line2D([0.30, 0.70], [0.49, 0.49], color=ACCENT, lw=2,
                              transform=fig.transFigure, alpha=alpha))
        fig.text(0.5, 0.40, "github.com/HiPerCAS/trifield", fontsize=18, color=INK, ha="center",
                 va="center", alpha=alpha)
        fig.text(0.5, 0.30, "pip install -e .   •   python examples/predict_fields.py",
                 fontsize=11, color=MUTED, ha="center", va="center", family="DejaVu Sans Mono",
                 alpha=alpha)
        fig.text(0.5, 0.16, "Popryho, Sadeghi, Partin-Vaisband  •  ICCAD 2026", fontsize=10,
                 color=MUTED, ha="center", va="center", alpha=alpha)
        frames.append((render(fig), 180 if alpha < 1 else 1300))
    return frames


# ----------------------------------------------------------------------------- main
def build_frames(models_pi, models_do, tag_pi: str, tag_do: str, data_dir: Path):
    snaps = [load_graph(data_dir / f"{DESIGN_1FIN}__{s}.pt") for s in SNAPSHOTS]
    g = snaps[-1]                                         # Vg = 5 V, Vd = 12 V corner
    g5 = load_graph(data_dir / f"{DESIGN_5FIN}__node1_des.pt")
    layer = Layer(g, axis=1, value=None, horiz=2, vert=0)      # x-z cross-section, densest y layer
    layer5 = Layer(g5, axis=0, value=0.0, horiz=2, vert=1)     # y-z top view at the GaN/AlGaN interface

    means_pi, sigmas_pi = zip(*(predict_ensemble(models_pi, s) for s in snaps))
    means_do = [predict_ensemble(models_do, s)[0] for s in snaps]
    mean5_pi, _ = predict_ensemble(models_pi, g5)
    mean5_do, _ = predict_ensemble(models_do, g5)
    stats = {"1fin_physics_informed": rmse_all(means_pi[-1], g), "1fin_data_only": rmse_all(means_do[-1], g),
             "5fin_physics_informed": rmse_all(mean5_pi, g5), "5fin_data_only": rmse_all(mean5_do, g5)}
    for k, v in stats.items():
        print(f"{k:24s} (Vg = 5 V, Vd = 12 V): RMSE psi {v[0]:.3f}  phi_n {v[1]:.3f}  phi_p {v[2]:.3f} V")
    vmin = min(float(s.y[:, 0].min()) for s in snaps)
    vmax = max(float(s.y[:, 0].max()) for s in snaps)

    frames = []
    frames += scene_title()
    frames += scene_design_to_graph(g, layer)
    frames += scene_message_passing(g, layer)
    frames += scene_pred_vs_tcad(g, layer, means_pi[-1], means_do[-1], stats["1fin_physics_informed"],
                                 stats["1fin_data_only"], tag_pi, tag_do)
    frames += scene_bias_sweep(snaps, layer, means_do, vmin, vmax, tag_do)
    frames += scene_uncertainty(g, layer, sigmas_pi[-1], means_pi[-1], tag_pi)
    frames += scene_size_transfer(g5, layer5, mean5_pi, mean5_do, stats["5fin_physics_informed"],
                                  stats["5fin_data_only"], g, tag_pi, tag_do)
    frames += scene_active_learning()
    frames += scene_closing("~4 orders of magnitude faster than the drift-diffusion solve")
    return frames, stats


def save_gif(frames, out: Path, colors: int = 256, speed: float = 1.0):
    imgs = [im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            for im, _ in frames]
    # speed < 1 slows the loop; short animation steps get a floor so motion stays readable
    durations = [int(max(d / speed, 250 if d < 300 else 0)) for _, d in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=durations, loop=0,
                 optimize=True, disposal=1)
    return out.stat().st_size


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output GIF (default: assets/framework.gif)")
    ap.add_argument("--device", default="auto", help='"auto" (cuda if available), "cuda" or "cpu"')
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="physics-informed ensemble checkpoints (globs allowed); "
                         "default: models/pignn_ens_seed{0,1}.pt")
    ap.add_argument("--models_dataonly", nargs="+", default=DEFAULT_MODELS_DATAONLY,
                    help="data-only ensemble checkpoints (globs allowed); default: models/dataonly_seed{0,1}.pt")
    ap.add_argument("--data_dir", default=DEFAULT_DATA, help="example graphs (default: examples/data)")
    ap.add_argument("--frames_dir", default=None, help="also dump every frame as PNG here")
    ap.add_argument("--speed", type=float, default=0.5,
                    help="playback speed factor (0.5 = twice as slow as the frame budget)")
    ap.add_argument("--colors", type=int, default=256, help="GIF palette size per frame")
    args = ap.parse_args(argv)

    ensembles, tags = [], []
    for pats in (args.models, args.models_dataonly):
        paths = sorted(p for pat in pats for p in (glob.glob(pat) or [pat]))
        ensembles.append(load_ensemble(paths, args.device))
        tags.append(short_tag(paths))
        print(f"loaded {len(ensembles[-1])} ensemble members: {', '.join(Path(p).name for p in paths)}")
    torch.manual_seed(0)
    frames, stats = build_frames(ensembles[0], ensembles[1], tags[0], tags[1], Path(args.data_dir))
    if args.frames_dir:
        fd = Path(args.frames_dir)
        fd.mkdir(parents=True, exist_ok=True)
        for i, (im, d) in enumerate(frames):
            im.save(fd / f"frame_{i:03d}_{d}ms.png")
    out = Path(args.out)
    size = save_gif(frames, out, args.colors, args.speed)
    total = sum(int(max(d / args.speed, 250 if d < 300 else 0)) for _, d in frames) / 1000
    shown = out.resolve().relative_to(ROOT) if out.resolve().is_relative_to(ROOT) else out
    print(f"wrote {shown}: {len(frames)} frames, {total:.1f} s loop, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
