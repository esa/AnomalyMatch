#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for the training subprocess size-histogram emission.

The subprocess emits the sampled unlabeled pool's source-size distribution over
the JSON-lines progress channel so the training screen can render it (PR3).
"""

import json
import os
import sys

import numpy as np
import pytest

import anomaly_match as am

# The subprocess imports a sibling module (prediction_utils) by name, so the
# scripts dir must be importable to load training_process here.
_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "subprocess_scripts",
)
sys.path.insert(0, _SCRIPTS_DIR)

import training_process as tp  # noqa: E402  (path inserted above)


class _FakeSource:
    def __init__(self, sizes, population=None, unit="pixel"):
        self._sizes = sizes
        # Population defaults to the sampled sizes so tests that only care about
        # the sampled series need not construct a separate candidate set.
        self._population = population if population is not None else sizes
        self._unit = unit

    def get_last_unlabeled_sizes(self):
        return self._sizes

    def get_last_unlabeled_population_sizes(self):
        return self._population

    def get_last_unlabeled_size_unit(self):
        return self._unit


def _read_lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_emits_size_histogram_when_sizes_present(tmp_path):
    progress = str(tmp_path / "progress.jsonl")
    cfg = am.get_default_cfg()
    cfg.cutana_stratify_source_size = True
    sizes = np.concatenate([np.full(100, 8.0), np.array([50.0, 120.0, 200.0])])
    # The candidate population is broader and more small-skewed than the sampled
    # pool; both are histogrammed onto the shared, population-derived edges.
    population = np.concatenate([np.full(1000, 8.0), np.array([50.0, 120.0, 200.0, 300.0])])

    tp._emit_size_histogram(progress, _FakeSource(sizes, population=population, unit="pixel"), cfg)

    lines = _read_lines(progress)
    assert len(lines) == 1
    msg = lines[0]
    assert msg["status"] == "size_histogram"
    # The flat region is cfg bins up to max_px; edges then continue past it to
    # cover the 300px tail source, so there are >= cfg bins and max_px is a boundary.
    n_edges = len(msg["edges"])
    assert n_edges == len(msg["sampled_counts"]) + 1
    assert n_edges == len(msg["population_counts"]) + 1
    assert n_edges >= cfg.cutana_size_stratify_bins + 1
    assert any(abs(e - cfg.cutana_size_stratify_max_px) < 1e-6 for e in msg["edges"])
    assert msg["edges"] == sorted(msg["edges"])  # monotonic
    assert msg["stratified"] is True
    assert msg["unit"] == "pixel"
    assert "log_x" not in msg
    # Edges span the full range, so every source lands in a bin and is counted.
    assert sum(msg["sampled_counts"]) == len(sizes)
    assert sum(msg["population_counts"]) == len(population)


@pytest.mark.parametrize("sizes", [None, np.array([])])
def test_emits_nothing_without_sizes(tmp_path, sizes):
    progress = str(tmp_path / "progress.jsonl")
    cfg = am.get_default_cfg()
    tp._emit_size_histogram(progress, _FakeSource(sizes), cfg)
    assert not os.path.exists(progress) or os.path.getsize(progress) == 0


def test_stratified_flag_reflects_config(tmp_path):
    progress = str(tmp_path / "progress.jsonl")
    cfg = am.get_default_cfg()
    cfg.cutana_stratify_source_size = False
    tp._emit_size_histogram(progress, _FakeSource(np.array([5.0, 6.0, 7.0])), cfg)
    assert _read_lines(progress)[0]["stratified"] is False
