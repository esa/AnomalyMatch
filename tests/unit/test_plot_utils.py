#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for the shared histogram renderer."""

import numpy as np

from anomaly_match_ui.utils.plot_utils import (
    render_histogram_png,
    render_size_stratification_png,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_renders_png_bytes():
    counts = np.array([3, 5, 2])
    edges = np.array([0.0, 0.33, 0.66, 1.0])
    png = render_histogram_png(counts, edges, bar_color="#00a1de")
    assert png[:8] == _PNG_MAGIC


def test_custom_xlabel_and_title_render():
    # The source-size histogram reuses the renderer with a diameter x-label and
    # a title to distinguish it from the score histogram in the shared slot.
    counts = np.array([10, 4, 1])
    edges = np.array([5.0, 70.0, 135.0, 200.0])
    png = render_histogram_png(
        counts,
        edges,
        bar_color="#84bd00",
        xlabel="Source diameter",
        title="Unlabeled size (uniform)",
    )
    assert png[:8] == _PNG_MAGIC
    # Titled render is a valid, non-trivial image.
    assert len(png) > 100


def test_single_bin_edges_do_not_crash():
    # A one-bin histogram gives edges where high == low after slicing; the
    # explicit left/mid/right tick logic must degrade to the two endpoints
    # rather than fail on a zero-width linspace.
    counts = np.array([7])
    edges = np.array([12.0, 12.0])
    png = render_histogram_png(counts, edges, bar_color="#84bd00")
    assert png[:8] == _PNG_MAGIC


def test_log_y_render():
    # The score histogram uses a log y-axis so the sparse high-score tail stays
    # legible against the sharp peak near zero.
    edges = np.linspace(0.0, 1.0, 31)
    counts = np.concatenate([np.array([5000]), np.full(29, 3)])
    png = render_histogram_png(counts, edges, bar_color="#00a1de", log_y=True)
    assert png[:8] == _PNG_MAGIC


def test_size_stratification_overlay_renders():
    # Population bars + sampled step line over shared log-spaced edges; the two
    # series are normalised independently so a large population and a small
    # sample stay comparable in shape.
    edges = np.logspace(np.log10(5), np.log10(200), 21)
    population_counts = np.concatenate([np.array([5000, 800]), np.full(18, 5)])
    sampled_counts = np.full(20, 10)
    png = render_size_stratification_png(
        edges,
        population_counts,
        sampled_counts,
        population_color="#003247",
        sampled_color="#ff7a00",
        unit="pixel",
        stratified=True,
    )
    assert png[:8] == _PNG_MAGIC
    assert len(png) > 100


def test_size_stratification_zero_population_does_not_crash():
    # An all-zero series must not divide-by-zero when normalising to a fraction.
    edges = np.logspace(np.log10(5), np.log10(200), 6)
    png = render_size_stratification_png(
        edges,
        np.zeros(5),
        np.array([1, 2, 3, 2, 1]),
        population_color="#003247",
        sampled_color="#ff7a00",
    )
    assert png[:8] == _PNG_MAGIC


def test_size_stratification_single_bin_edges_do_not_crash():
    # The overlay always uses a log x-axis; a degenerate single-bin range
    # (high == low) must degrade to the two endpoints rather than fail on a
    # zero-width geomspace.
    edges = np.array([12.0, 12.0])
    png = render_size_stratification_png(
        edges,
        np.array([5]),
        np.array([2]),
        population_color="#003247",
        sampled_color="#ff7a00",
    )
    assert png[:8] == _PNG_MAGIC
