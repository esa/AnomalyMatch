#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Thread-safety regression test for the batched Cutana cache decoder.

Preview decode runs in a background thread.  If the user edits the
normalisation widget mid-decode, the UI thread replaces ``cfg.fitsbolt_cfg``.
``decode_cutana_raw_images`` must process the whole batch with one consistent
config; otherwise later cutouts come out with a different ``n_output_channels``
and ``np.stack`` raises ``"all input arrays must have the same shape"``.
"""

import fitsbolt
import numpy as np
import pytest

import anomaly_match as am
from anomaly_match.data_io import container_loaders
from anomaly_match.data_io.load_images import get_fitsbolt_config


@pytest.fixture
def asinh_cfg():
    """Factory building an ASINH config with a given output-channel layout.

    Returns a callable so each test can request the channel count and
    combination matrix it needs (including the rival config used to simulate a
    concurrent edit).
    """

    def _make(n_output, channel_combination=None):
        cfg = am.get_default_cfg()
        cfg.normalisation.normalisation_method = fitsbolt.NormalisationMethod.ASINH
        cfg.normalisation.n_output_channels = n_output
        cfg.normalisation.image_size = [16, 16]
        if channel_combination is not None:
            cfg.normalisation.channel_combination = channel_combination
        return get_fitsbolt_config(cfg)

    return _make


def test_decode_survives_concurrent_fitsbolt_cfg_swap(asinh_cfg, monkeypatch):
    """A mid-batch ``cfg.fitsbolt_cfg`` swap must not change the batch's shape."""
    cfg = asinh_cfg(3, [[1.0, 0.66, 0.33, 0.0], [0.0, 0.0, 0.33, 0.66], [1.0, 0.0, 0.0, 0.0]])
    original_fb_cfg = cfg.fitsbolt_cfg
    before_n_out = original_fb_cfg.n_output_channels

    # Uniform 4-band raw cutouts, as the labeled cache stores them.
    raw = [np.random.RandomState(i).rand(24, 24, 4).astype(np.float32) for i in range(8)]

    # A rival config with a different output-channel count — installing it
    # part-way through the loop used to flip later cutouts to 3 channels.
    rival = asinh_cfg(4)

    real = container_loaders.process_single_wrapper
    calls = 0

    def swapping_wrapper(image, passed_cfg, desc="x"):
        nonlocal calls
        calls += 1
        if calls == 3:
            cfg.fitsbolt_cfg = rival.fitsbolt_cfg  # simulate the UI thread's edit
        return real(image, passed_cfg, desc=desc)

    monkeypatch.setattr(container_loaders, "process_single_wrapper", swapping_wrapper)

    out = container_loaders.decode_cutana_raw_images(raw, cfg)

    shapes = {o.shape for o in out}
    assert shapes == {(16, 16, 3)}, f"batch shape diverged under concurrent edit: {shapes}"
    # The decode must not mutate the caller's config object either.
    assert original_fb_cfg.n_output_channels == before_n_out


def test_decode_does_not_mutate_shared_config(asinh_cfg):
    """Decoding leaves ``cfg.fitsbolt_cfg`` untouched (no transient state leak)."""
    cfg = asinh_cfg(3, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    before_n_out = cfg.fitsbolt_cfg.n_output_channels
    raw = [np.random.RandomState(i).rand(20, 20, 4).astype(np.float32) for i in range(4)]

    out = container_loaders.decode_cutana_raw_images(raw, cfg)

    assert {o.shape for o in out} == {(16, 16, 3)}
    assert cfg.fitsbolt_cfg.n_output_channels == before_n_out
    assert cfg.fitsbolt_cfg is not None
