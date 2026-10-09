#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Plotting utilities extracted from UI screens for reuse and testability."""

from __future__ import annotations

import io

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _apply_range_xticks(ax: Axes, edges: np.ndarray, *, log_x: bool) -> None:
    """Put legible left/middle/right x-ticks on a small dark histogram axis.

    Matplotlib's defaults on a small dark figure (and log's ``10**n`` locator)
    otherwise leave the bars with no readable scale, so we place three explicit
    ticks spanning the bin range and format them to the value magnitude.

    Args:
        ax: Target axes.
        edges: Histogram bin edges.
        log_x: Whether the x-axis is logarithmic (uses geometric tick spacing).
    """
    low, high = float(edges[0]), float(edges[-1])
    if high > low:
        tick_positions = np.geomspace(low, high, 3) if log_x else np.linspace(low, high, 3)
    else:
        tick_positions = np.array([low, high])
    if high >= 10:
        tick_labels = [f"{t:.0f}" for t in tick_positions]
    elif high >= 1:
        tick_labels = [f"{t:.1f}" for t in tick_positions]
    else:
        tick_labels = [f"{t:.2f}" for t in tick_positions]

    if log_x:
        ax.set_xscale("log")
        # A custom major locator overrides log's decade minor ticks, which would
        # otherwise reintroduce the illegible clutter this tick set replaces.
        ax.minorticks_off()
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)


def _style_dark_axes(ax: Axes, *, xlabel: str, ylabel: str, title: str | None) -> None:
    """Apply the shared dark-theme styling used by both histogram renderers.

    Args:
        ax: Target axes.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Optional axes title.
    """
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    # Grid sits behind the bars for readability without obscuring the counts.
    ax.set_axisbelow(True)
    ax.grid(True, color="#444", linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333")


def _figure_to_png(fig: Figure, dpi: int) -> bytes:
    """Serialise a figure to PNG bytes on the black UI background.

    Args:
        fig: The figure to render.
        dpi: Output resolution.

    Returns:
        PNG image as bytes.
    """
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black", dpi=dpi)
    buf.seek(0)
    return buf.getvalue()


def render_histogram_png(
    counts: np.ndarray,
    edges: np.ndarray,
    bar_color: str,
    *,
    figsize: tuple[float, float] = (3.0, 2.0),
    dpi: int = 100,
    xlabel: str = "Score",
    title: str | None = None,
    log_y: bool = False,
) -> bytes:
    """Render a histogram as PNG bytes.

    Used for the linear-x score histogram; the source-size distribution has its
    own overlay renderer (:func:`render_size_stratification_png`).

    Args:
        counts: Bin counts from a histogram.
        edges: Bin edges (length ``len(counts) + 1``).
        bar_color: CSS colour string for the bars.
        figsize: Figure size in inches.
        dpi: Output resolution.
        xlabel: X-axis label (defaults to ``"Score"`` for the score histogram).
        title: Optional axes title.
        log_y: Use a logarithmic y-axis — the score distribution is sharply
            peaked, so a linear count axis buries the rare high-score tail (where
            the anomalies live) in a flat line at the bottom.

    Returns:
        PNG image as bytes.
    """
    matplotlib.use("Agg")
    fig = Figure(figsize=figsize, facecolor="black")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    ax.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        color=bar_color,
        edgecolor="none",
        align="edge",
    )
    _apply_range_xticks(ax, edges, log_x=False)
    if log_y:
        ax.set_yscale("log")
        # A count of 1 must sit above the axis floor to be visible; 0.5 keeps the
        # single-count bars legible without leaving a large empty gap below them.
        ax.set_ylim(bottom=0.5)
    _style_dark_axes(ax, xlabel=xlabel, ylabel="Count", title=title)
    return _figure_to_png(fig, dpi)


def render_size_stratification_png(
    edges: np.ndarray,
    population_counts: np.ndarray,
    sampled_counts: np.ndarray,
    *,
    population_color: str,
    sampled_color: str,
    unit: str = "pixel",
    stratified: bool = False,
    figsize: tuple[float, float] = (3.0, 2.0),
    dpi: int = 100,
) -> bytes:
    """Overlay the sampled unlabeled pool's sizes on the tile-population sizes.

    The filled bars are the candidate population (every source in the drawn
    tiles); the step line is the sampled unlabeled pool.  Both are normalised to
    a per-bin fraction so their shapes are comparable despite the population
    dwarfing the sample in absolute count — with stratification on, the sampled
    line should be visibly flatter (~uniform in log size) than the small-skewed
    population, which is the visual confirmation that stratification worked.

    Args:
        edges: Shared log-spaced bin edges (length ``len(counts) + 1``).
        population_counts: Per-bin counts of the tile population.
        sampled_counts: Per-bin counts of the sampled unlabeled pool.
        population_color: CSS colour for the population bars.
        sampled_color: CSS colour for the sampled step line.
        unit: Diameter unit for the x-axis label (``"pixel"`` / ``"arcsec"``).
        stratified: Whether size stratification was applied (annotates the title).
        figsize: Figure size in inches.
        dpi: Output resolution.

    Returns:
        PNG image as bytes.
    """
    matplotlib.use("Agg")
    population_counts = np.asarray(population_counts, dtype=float)
    sampled_counts = np.asarray(sampled_counts, dtype=float)

    def _fraction(counts: np.ndarray) -> np.ndarray:
        total = counts.sum()
        return counts / total if total > 0 else counts

    population_fraction = _fraction(population_counts)
    sampled_fraction = _fraction(sampled_counts)

    fig = Figure(figsize=figsize, facecolor="black")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    ax.bar(
        edges[:-1],
        population_fraction,
        width=np.diff(edges),
        color=population_color,
        edgecolor="none",
        align="edge",
        label="In tiles",
    )
    # A step outline (aligned to the same bin edges) reads clearly on top of the
    # filled population bars without a second, occluding set of bars.
    ax.stairs(sampled_fraction, edges, color=sampled_color, linewidth=1.8, label="Sampled")
    _apply_range_xticks(ax, edges, log_x=True)
    title = "Sampled vs tiles (stratified)" if stratified else "Sampled vs tiles"
    _style_dark_axes(ax, xlabel=f"Source diameter ({unit})", ylabel="Fraction", title=title)
    ax.legend(
        facecolor="black",
        edgecolor="#333",
        labelcolor="white",
        fontsize=7,
        loc="upper right",
    )
    return _figure_to_png(fig, dpi)
