#!/usr/bin/env python3
"""OlympiadBench motivation figure: truncation gap and token savings."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[2]
if "PLWS_ROOT" in __import__("os").environ:
    ROOT = Path(__import__("os").environ["PLWS_ROOT"]).resolve()
CACHE = ROOT / "tmp/plws_step_probe/r1_7b_count_bias_s42_token_series_post.json"
FULL_META = ROOT / "tmp/plws_step_probe/count_bias/rho_0p98/r1_7b/fullcot_seed42_series.json"
OUT_DIR = ROOT / "tmp/plws_step_probe"


def first_cross(xs: list[float], ys: list[float], level: float) -> float | None:
    for index in range(1, len(ys)):
        if ys[index - 1] < level <= ys[index]:
            span = ys[index] - ys[index - 1]
            weight = 0.0 if span == 0 else (level - ys[index - 1]) / span
            return xs[index - 1] + weight * (xs[index] - xs[index - 1])
    return None


def outline_text(
    ax,
    x,
    y,
    lines,
    size=10,
    color="#000000",
    align="center",
    ha="left",
    weights=None,
):
    """Draw text as glyph outlines so a delta stays vector without a CID font."""
    if weights is None:
        weights = ["normal"] * len(lines)
    leading = size * 1.28
    paths = []
    for index, (line, weight) in enumerate(zip(lines, weights, strict=True)):
        prop = FontProperties(family="STIXGeneral", size=size, weight=weight)
        path = TextPath((0, 0), line, size=size, prop=prop)
        paths.append(path.transformed(Affine2D().translate(0, -index * leading)))
    boxes = [path.get_extents() for path in paths]
    if align == "top":
        y_edge = boxes[0].y1
    elif align == "first":
        y_edge = (boxes[0].y0 + boxes[0].y1) / 2
    else:
        y_edge = (min(box.y0 for box in boxes) + max(box.y1 for box in boxes)) / 2
    anchor = ax.transData.transform((x, y))
    scale = ax.figure.dpi / 72.0
    for path, box in zip(paths, boxes, strict=True):
        if ha == "right":
            x_edge = box.x1
        elif ha == "center":
            x_edge = (box.x0 + box.x1) / 2
        else:
            x_edge = box.x0
        placed = path.transformed(
            Affine2D().translate(-x_edge, -y_edge).scale(scale).translate(anchor[0], anchor[1])
        )
        data_path = placed.transformed(ax.transData.inverted())
        ax.add_patch(
            PathPatch(data_path, facecolor=color, edgecolor="none", lw=0, zorder=7)
        )


def curved_extension(xs: list[float], ys: list[float], level: float, count: int = 48):
    """Stay with the flat tail, then bend up to ``level``."""
    import numpy as np

    x0 = float(xs[-1])
    y0 = float(ys[-1])
    dx = 0.82
    steps = np.linspace(0.0, 1.0, count)[1:]
    extra_x = [x0 + float(t) * dx for t in steps]
    extra_y = [y0 + float(level - y0) * float(t) ** 2.6 for t in steps]
    return extra_x, extra_y, x0 + dx


def text_block_size(lines, size) -> tuple[float, float]:
    """Return width and height in points of a two-line label."""
    leading = size * 1.28
    width = 0.0
    top = 0.0
    bottom = 0.0
    for index, line in enumerate(lines):
        weight = "bold" if index == 0 else "normal"
        prop = FontProperties(family="STIXGeneral", size=size, weight=weight)
        box = TextPath((0, 0), line, size=size, prop=prop).get_extents()
        width = max(width, box.width)
        shifted_top = box.y1 - index * leading
        shifted_bottom = box.y0 - index * leading
        top = max(top, shifted_top)
        bottom = min(bottom, shifted_bottom) if index else shifted_bottom
    return width, top - bottom


def conclusion_box(ax, x, y, width, height, lines, size, ink, scale):
    """Draw a rounded callout. ``scale`` shrinks frame and type together."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad={0.04 * scale:.4f},rounding_size={0.13 * scale:.4f}",
            facecolor="#FAFAFA",
            edgecolor="#3A3A3A",
            linewidth=2.8 * scale,
            zorder=6,
        )
    )
    outline_text(
        ax,
        x + width / 2,
        y + height / 2,
        lines,
        size=size,
        color=ink,
        ha="center",
        weights=["bold", "normal"],
    )


def render(spec: dict, full_x, full_y, bias_x, bias_y, final, start, gap, fewer) -> None:
    plt.rcParams.update(
        {
            "pdf.use14corefonts": True,
            "ps.useafm": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Times"],
            "mathtext.fontset": "stix",
            "font.size": spec["font"],
            "axes.labelsize": spec["label"],
            "xtick.labelsize": spec["tick_font"],
            "ytick.labelsize": spec["tick_font"],
            "axes.linewidth": spec["axes_lw"],
        }
    )
    blue = "#3478A8"
    teal = "#22A38F"
    reference = "#668AA3"
    danger = "#F40000"
    benefit = "#F40000"
    ink = "#000000"
    # Real post-</think> trajectories only; no drawn extension.
    meet_x = max(float(full_x[-1]), float(bias_x[-1]))
    fig, ax = plt.subplots(figsize=spec["figsize"])
    ax.plot(full_x, full_y, color=blue, lw=spec["curve_lw"], label="Vanilla CoT", zorder=3)
    ax.plot(bias_x, bias_y, color=teal, lw=spec["curve_lw"], label="ATP Suppression", zorder=4)
    ax.axhline(
        final,
        color=reference,
        ls="--",
        lw=spec["ref_lw"],
        label="Vanilla CoT (Final Acc)",
        zorder=2,
    )
    ax.axhline(
        start,
        color="#8A4B3A",
        ls=(0, (1.2, 1.4)),
        lw=spec["ref_lw"],
        label="Truncation after agreement",
        zorder=1,
    )

    gap_x = 0.28
    tick = 0.20
    ax.annotate(
        "",
        xy=(gap_x, final),
        xytext=(gap_x, start),
        arrowprops=dict(
            arrowstyle="<->",
            color=danger,
            lw=spec["leader_lw"],
            mutation_scale=spec["leader_head"],
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )
    cap = dict(color=danger, linestyle="solid", linewidth=spec["leader_lw"], zorder=7)
    ax.plot([gap_x - tick, gap_x + tick], [start, start], **cap)
    ax.plot([gap_x - tick, gap_x + tick], [final, final], **cap)

    bias_end_x, bias_end_y = float(bias_x[-1]), float(bias_y[-1])
    full_end_x, full_end_y = float(full_x[-1]), float(full_y[-1])
    span_y = 55.15
    over = 0.22
    link = dict(
        color=benefit,
        linestyle="solid",
        linewidth=spec["leader_lw"],
        solid_capstyle="butt",
        zorder=5,
    )
    ax.plot([bias_end_x, bias_end_x], [bias_end_y, span_y - over], **link)
    ax.plot([full_end_x, full_end_x], [full_end_y, span_y - over], **link)
    ax.annotate(
        "",
        xy=(full_end_x, span_y),
        xytext=(bias_end_x, span_y),
        arrowprops=dict(
            arrowstyle="<->",
            color=benefit,
            lw=spec["leader_lw"],
            mutation_scale=spec["leader_head"],
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )

    ax.set_xlim(0, meet_x + 0.28)
    ax.set_ylim(44, 58.5)
    ax.set_yticks(spec["yticks"])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.set_xticklabels(["0", "1k", "2k", "3k", "4k", "5k", "6k", "7k"])
    ax.set_xlabel("Reasoning Tokens after Agreement")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(color="#D8E0E3", lw=spec["grid_lw"])
    ax.tick_params(labelsize=spec["tick_font"], width=spec["tick_width"], length=spec["tick_len"])
    ax.set_axisbelow(True)
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=1,
        edgecolor="#bbbbbb",
        fontsize=spec["legend"],
        borderpad=spec["legend_pad"],
        handlelength=spec["handle"],
        labelspacing=0.25,
    )
    if spec["layout"] == "column":
        fig_w, fig_h = spec["figsize"]
        pad_in = 2.0 / 25.4
        fig.subplots_adjust(
            left=(pad_in + 0.36) / fig_w,
            right=1 - pad_in / fig_w,
            bottom=(pad_in + 0.32) / fig_h,
            top=1 - (pad_in + 0.02) / fig_h,
        )
    else:
        fig.tight_layout()
    fig.canvas.draw()
    # Right callout matches the legend width and sits just above it.
    # Left callout is the same size; the gap between the two stays 0.16.
    box_lines = (
        ["(1) Truncation gap", f"(ΔAcc = −{gap:.1f}%)"],
        ["(2) Token reduction", f"(ΔToken = −{fewer:.1f}%)"],
    )
    ref_size = 21.0
    ref_w, ref_h = 0.0, 0.0
    for lines in box_lines:
        block_w, block_h = text_block_size(lines, ref_size)
        ref_w = max(ref_w, block_w)
        ref_h = max(ref_h, block_h)
    origin = ax.transData.transform((0.0, 0.0))
    unit_x = ax.transData.transform((1.0, 0.0))
    unit_y = ax.transData.transform((0.0, 1.0))
    pt_per_x = abs(unit_x[0] - origin[0]) * 72.0 / fig.dpi
    pt_per_y = abs(unit_y[1] - origin[1]) * 72.0 / fig.dpi
    natural_w = (ref_w / 0.90) / pt_per_x
    natural_h = (ref_h / 0.78) / pt_per_y
    legend_px = legend.get_window_extent()
    inv = ax.transData.inverted()
    legend_left, legend_bottom = inv.transform((legend_px.x0, legend_px.y0))
    legend_right, legend_top = inv.transform((legend_px.x1, legend_px.y1))
    legend_w = legend_right - legend_left
    scale = legend_w / natural_w
    pad = 0.04 * scale
    box_w = legend_w - 2 * pad
    box_h = natural_h * scale
    note_size = ref_size * scale
    gap_box = 0.16
    right_box_x = legend_left + pad
    left_box_x = right_box_x - gap_box - box_w - 2 * pad
    box_y = legend_top + 0.35 + pad
    conclusion_box(ax, left_box_x, box_y, box_w, box_h, box_lines[0], size=note_size, ink=ink, scale=scale)
    conclusion_box(ax, right_box_x, box_y, box_w, box_h, box_lines[1], size=note_size, ink=ink, scale=scale)
    leader = dict(
        arrowstyle="-|>",
        color="#3A3A3A",
        lw=spec["leader_lw"],
        mutation_scale=spec["leader_head"],
        shrinkA=0,
        shrinkB=0,
    )
    tip_y = box_y + box_h + 0.04 * scale
    ax.annotate(
        "",
        xy=(left_box_x + box_w / 2, tip_y),
        xytext=(gap_x, final - 2.4),
        arrowprops={**leader, "connectionstyle": "arc3,rad=0.0"},
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(right_box_x + box_w / 2, tip_y),
        xytext=((bias_end_x + full_end_x) / 2, span_y),
        arrowprops={**leader, "connectionstyle": "arc3,rad=0.0"},
        zorder=6,
    )
    if spec["layout"] == "column":
        fig.savefig(spec["pdf"], format="pdf")
        fig.savefig(spec["png"], dpi=220)
    else:
        fig.savefig(spec["pdf"], format="pdf", bbox_inches="tight")
        fig.savefig(spec["png"], dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(
        spec["pdf"],
        f"meet_x={meet_x:.3f}",
        f"note={note_size:.2f}",
        f"scale={scale:.3f}",
        f"box=({left_box_x:.2f},{box_y:.2f},{box_w:.2f},{box_h:.2f})",
        f"right_x={right_box_x:.2f}",
        f"legend=({legend_left:.2f},{legend_top:.2f},{legend_w:.2f})",
    )


def main() -> None:
    row = json.loads(CACHE.read_text())["olympiadbench"]
    meta = json.loads(FULL_META.read_text())["datasets"]["olympiadbench"]
    full_x = row["full_x"]
    full_y = row["full_y"]
    bias_x = row["count_bias_x"]
    bias_y = row["count_bias_y"]
    # Same 633-question denominator as the curves (not the full 675-set).
    final = float(meta["window_orig_acc"])
    start = float(bias_y[0])
    gap = round(final, 1) - round(start, 1)
    level = 56.0
    left = first_cross(bias_x, bias_y, level)
    right = first_cross(full_x, full_y, level)
    if left is None or right is None:
        raise SystemExit("iso-accuracy crossings missing")
    fewer = 100.0 * (1.0 - left / right)
    scale = 1000.0
    full_x = [value / scale for value in full_x]
    bias_x = [value / scale for value in bias_x]
    left /= scale
    right /= scale
    series = (full_x, full_y, bias_x, bias_y, final, start, gap, fewer)
    render(
        {
            "figsize": (8.4, 5.3),
            "font": 15.0,
            "label": 22.0,
            "tick_font": 18.5,
            "axes_lw": 2.2,
            "curve_lw": 4.0,
            "ref_lw": 2.6,
            "arrow_lw": 4.2,
            "arrow_head": 24,
            "cap_lw": 3.8,
            "link_lw": 3.4,
            "leader_lw": 3.2,
            "leader_head": 22,
            "grid_lw": 1.15,
            "tick_width": 1.6,
            "tick_len": 5.5,
            "note": 15.0,
            "legend": 13.5,
            "legend_pad": 0.5,
            "handle": 2.3,
            "yticks": [44, 46, 48, 50, 52, 54, 56, 58],
            "layout": "wide",
            "pdf": OUT_DIR / "olympiad_post_agreement_motivation_wide.pdf",
            "png": OUT_DIR / "olympiad_post_agreement_motivation_wide.png",
        },
        *series,
    )


if __name__ == "__main__":
    main()
