#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for skipping FITS extensions with zero channel_combination weight.

A band that every output channel weights at zero contributes nothing to the
final cutout, so reading it is wasted I/O (each band is a separate windowed
read).  The streaming prediction/training paths therefore drop such bands at
extraction and the combine matmul drops the matching all-zero matrix columns.

The central guarantee these tests pin down is that pruning **never changes the
result and never drops data the output depends on**: the pruned pipeline is
bit-identical to the full one, and a band is only ever skipped when the matrix
already ignores it.  The labeled-cache/preview path keeps every band so the
combination can be retuned without re-extracting.
"""

import numpy as np
import pytest
import torch
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

import anomaly_match as am
from anomaly_match.data_io.container_loaders import apply_channel_combination_to_cutana_batch
from anomaly_match.data_io.load_images import (
    _drop_unused_combination_columns,
    get_fitsbolt_config,
)
from anomaly_match.datasets.cutana_source import (
    _prune_unused_extensions,
    build_cutana_orchestrator_config,
)
from prediction_utils import cutana_batch_to_model_tensor_gpu

EXTENSIONS = ["VIS", "NIR-H", "NIR-Y", "NIR-J"]


def _cfg(channel_combination, n_output):
    """Minimal config carrying a channel_combination, for the combine helpers."""
    cfg = am.get_default_cfg()
    cfg.normalisation.n_output_channels = n_output
    cfg.normalisation.channel_combination = channel_combination
    return cfg


def _build_cfg(channel_combination, n_output, extensions=EXTENSIONS):
    """Config ready for ``build_cutana_orchestrator_config``.

    A string ``fits_extension`` list avoids a catalogue read, and
    ``get_fitsbolt_config`` populates the ``fitsbolt_cfg`` the builder requires.
    """
    cfg = am.get_default_cfg()
    cfg.normalisation.fits_extension = list(extensions)
    cfg.normalisation.n_output_channels = n_output
    cfg.normalisation.channel_combination = channel_combination
    return get_fitsbolt_config(cfg)


class TestPruneUnusedExtensions:
    """``_prune_unused_extensions`` drops only bands the matrix already ignores."""

    def test_drops_zero_weight_band_preserving_order(self):
        # Grayscale from VIS only — the three NIR bands are dead weight.
        names, mask = _prune_unused_extensions(_cfg([[1, 0, 0, 0]], 1), EXTENSIONS)
        assert names == ["VIS"]
        # Mask spans the original band order so the caller can subset per-band params.
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_keeps_band_used_by_any_output_channel(self):
        # VIS feeds channel 0, NIR-Y feeds channel 1; NIR-H/NIR-J unused.
        names, mask = _prune_unused_extensions(_cfg([[1, 0, 0, 0], [0, 0, 1, 0]], 2), EXTENSIONS)
        assert names == ["VIS", "NIR-Y"]
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_keeps_all_when_every_band_weighted(self):
        # The real VIS+NIR grayscale blend from the field report — all 4 bands
        # contribute, so nothing may be dropped.
        cc = [[4 / 7, 1 / 7, 1 / 7, 1 / 7]]
        names, mask = _prune_unused_extensions(_cfg(cc, 1), EXTENSIONS)
        assert names == EXTENSIONS
        assert mask is None

    def test_no_combination_keeps_all(self):
        assert _prune_unused_extensions(_cfg(None, 1), EXTENSIONS) == (EXTENSIONS, None)

    def test_all_zero_matrix_keeps_all(self):
        # Degenerate matrix: refuse to prune down to nothing.
        assert _prune_unused_extensions(_cfg([[0, 0, 0, 0]], 1), EXTENSIONS) == (EXTENSIONS, None)

    def test_matrix_width_mismatch_raises(self):
        # A matrix that doesn't line up with the bands can't be paired with them,
        # so any combine would mix the wrong bands — a broken invariant, not an
        # intentional no-prune case. Fail hard rather than guess.
        with pytest.raises(ValueError, match=r"2 column\(s\).*4 FITS extension\(s\)"):
            _prune_unused_extensions(_cfg([[1, 0]], 1), EXTENSIONS)


class TestBuildConfigPruning:
    """End-to-end through the orchestrator config: pruning must stay opt-in and
    keep ``selected_extensions``/weights/fitsbolt band count mutually consistent."""

    def test_prune_drops_unused_selected_extensions(self):
        ccfg = build_cutana_orchestrator_config(
            "/fake/cat.parquet", _build_cfg([[1, 0, 0, 0]], 1), prune_unused_bands=True
        )
        assert [e["name"] for e in ccfg.selected_extensions] == ["VIS"]
        # Identity weights, fitsbolt band counts and fits_extensions must all
        # collapse to the single surviving band — no stale 4-band leftovers.
        assert list(ccfg.channel_weights) == ["VIS"]
        assert ccfg.external_fitsbolt_cfg.n_output_channels == 1
        assert ccfg.external_fitsbolt_cfg.n_expected_channels == 1
        assert ccfg.fits_extensions == ["VIS"]

    def test_default_keeps_all_bands(self):
        # Off by default: the labeled cache and preview paths rely on this so the
        # combination can be retuned later without re-extracting missing bands.
        ccfg = build_cutana_orchestrator_config("/fake/cat.parquet", _build_cfg([[1, 0, 0, 0]], 1))
        assert [e["name"] for e in ccfg.selected_extensions] == EXTENSIONS

    def test_prune_keeps_all_when_every_band_weighted(self):
        ccfg = build_cutana_orchestrator_config(
            "/fake/cat.parquet",
            _build_cfg([[4 / 7, 1 / 7, 1 / 7, 1 / 7]], 1),
            prune_unused_bands=True,
        )
        assert [e["name"] for e in ccfg.selected_extensions] == EXTENSIONS


class TestPrunedPerBandNormalisation:
    """Pruning must keep each surviving band's per-band normalisation params
    identical to the unpruned run — a non-prefix prune must subset the params by
    the kept-band mask, not truncate them to the first ``n``."""

    def test_asinh_params_match_full_band_slice_for_non_prefix_prune(self):
        # Keep bands 1 (NIR-H) and 3 (NIR-J) — a non-prefix selection, the case a
        # truncating resize gets wrong.  Distinct per-band ASINH params expose it.
        cfg = am.get_default_cfg()
        cfg.normalisation.fits_extension = EXTENSIONS
        cfg.normalisation.n_output_channels = 2
        cfg.normalisation.channel_combination = [[0, 1, 0, 0], [0, 0, 0, 1]]
        cfg.normalisation.normalisation_method = NormalisationMethod.ASINH
        cfg.normalisation.norm_asinh_scale = [10.0, 20.0]
        cfg.normalisation.norm_asinh_clip = [1.0, 2.0]
        cfg = get_fitsbolt_config(cfg)

        full = build_cutana_orchestrator_config("/fake/cat.parquet", cfg)
        pruned = build_cutana_orchestrator_config("/fake/cat.parquet", cfg, prune_unused_bands=True)

        for attr in ("asinh_scale", "asinh_clip"):
            full_params = list(full.external_fitsbolt_cfg.normalisation[attr])
            pruned_params = list(pruned.external_fitsbolt_cfg.normalisation[attr])
            # The two surviving bands must carry exactly their full-run params.
            assert pruned_params == [full_params[1], full_params[3]]


class TestDropUnusedCombinationColumns:
    """The matmul-side realignment that pairs with extraction pruning."""

    def test_drops_zero_columns_to_match_band_count(self):
        cc = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(_drop_unused_combination_columns(cc, 2), [[1.0, 1.0]])

    def test_raises_when_alignment_impossible(self):
        # Two non-zero columns can't be made to match a three-band batch — a
        # genuine extraction/combination disagreement, so fail hard.
        cc = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="disagree on which bands"):
            _drop_unused_combination_columns(cc, 3)


class TestPrunedCombineEquivalence:
    """The result must be identical whether or not unused bands were dropped —
    the proof that pruning never silently changes the model's input."""

    # Zero columns at indices 1 and 3, so extraction keeps bands [VIS, NIR-Y].
    CHANNEL_COMBINATION = [[1.0, 0.0, 1.0, 0.0]]

    def _batches(self):
        rng = np.random.RandomState(0)
        full = rng.randint(0, 256, (4, 8, 8, 4), dtype=np.uint8)
        pruned = full[..., [0, 2]]  # the bands extraction would keep, in order
        return full, pruned

    def test_cpu_batch_combine_matches(self):
        cfg = _cfg(self.CHANNEL_COMBINATION, n_output=1)
        full, pruned = self._batches()
        out_full = apply_channel_combination_to_cutana_batch(full, cfg)
        out_pruned = apply_channel_combination_to_cutana_batch(pruned, cfg)
        np.testing.assert_array_equal(out_full, out_pruned)

    def test_gpu_path_tensor_matches(self):
        cfg = _cfg(self.CHANNEL_COMBINATION, n_output=1)
        full, pruned = self._batches()
        device = torch.device("cpu")
        t_full = cutana_batch_to_model_tensor_gpu(full, cfg, device=device)
        t_pruned = cutana_batch_to_model_tensor_gpu(pruned, cfg, device=device)
        torch.testing.assert_close(t_full, t_pruned)
