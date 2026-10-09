#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the parallel labeled-cache decode (issue #500).

Decoding a cached cutout is dominated by the resize anti-aliasing gaussian
filter, a GIL-releasing C routine, so :func:`decode_cutana_raw_images` fans the
batch across the pod's cores.  The parallel path must produce *bit-identical*
output to the serial path — these tests pin that invariant and the thread-count
sizing logic.
"""

import fitsbolt
import numpy as np
import pytest

import anomaly_match as am
from anomaly_match.data_io import container_loaders
from anomaly_match.data_io.container_loaders import (
    _decode_thread_count,
    decode_cutana_raw_images,
)
from anomaly_match.data_io.load_images import get_fitsbolt_config


@pytest.fixture
def asinh_cfg():
    """Build an ASINH 4-band -> 3-channel config matching the labeled cache."""
    cfg = am.get_default_cfg()
    cfg.normalisation.normalisation_method = fitsbolt.NormalisationMethod.ASINH
    cfg.normalisation.n_output_channels = 3
    cfg.normalisation.image_size = [16, 16]
    cfg.normalisation.channel_combination = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    return get_fitsbolt_config(cfg)


class TestDecodeThreadCount:
    def test_capped_at_batch_size(self, monkeypatch):
        # Pretend the pod has 8 cores; a 3-image batch must not over-spawn.
        monkeypatch.setattr(container_loaders, "_effective_cpu_count", lambda: 8)
        assert _decode_thread_count(3) == 3

    def test_capped_at_effective_cpu_count(self, monkeypatch):
        monkeypatch.setattr(container_loaders, "_effective_cpu_count", lambda: 4)
        assert _decode_thread_count(1000) == 4

    def test_single_image_is_serial(self, monkeypatch):
        monkeypatch.setattr(container_loaders, "_effective_cpu_count", lambda: 8)
        assert _decode_thread_count(1) == 1

    def test_queries_and_caches_effective_cpu_count(self, monkeypatch):
        """It asks cutana's SystemMonitor once (the Kubernetes-aware count) and
        memoises the result via the lru_cache."""
        container_loaders._effective_cpu_count.cache_clear()
        calls = {"n": 0}

        class _FakeMonitor:
            def get_effective_cpu_count(self):
                calls["n"] += 1
                return 4

        monkeypatch.setattr(container_loaders, "SystemMonitor", _FakeMonitor)

        try:
            assert _decode_thread_count(1000) == 4
            # Cached: a second call must not construct/query the monitor again.
            assert _decode_thread_count(1000) == 4
            assert calls["n"] == 1
            assert container_loaders._effective_cpu_count() == 4
        finally:
            # Don't leak the fake's cached 4 into other tests.
            container_loaders._effective_cpu_count.cache_clear()


class TestParallelMatchesSerial:
    """The whole point: threading must not perturb a single output pixel."""

    def _decode_with_workers(self, raws, cfg, workers, monkeypatch):
        monkeypatch.setattr(container_loaders, "_effective_cpu_count", lambda: workers)
        return decode_cutana_raw_images(raws, cfg)

    def test_identical_output(self, asinh_cfg, monkeypatch):
        raws = [np.random.RandomState(i).rand(24, 24, 4).astype(np.float32) for i in range(20)]

        serial = self._decode_with_workers(raws, asinh_cfg, 1, monkeypatch)
        parallel = self._decode_with_workers(raws, asinh_cfg, 8, monkeypatch)

        assert len(serial) == len(parallel) == 20
        for a, b in zip(serial, parallel):
            assert np.array_equal(a, b)

    def test_order_is_preserved(self, asinh_cfg, monkeypatch):
        # Distinct constant-fill cutouts so an out-of-order result is obvious.
        raws = [np.full((24, 24, 4), float(i), dtype=np.float32) for i in range(12)]

        serial = self._decode_with_workers(raws, asinh_cfg, 1, monkeypatch)
        parallel = self._decode_with_workers(raws, asinh_cfg, 8, monkeypatch)

        for a, b in zip(serial, parallel):
            assert np.array_equal(a, b)

    def test_empty_batch(self, asinh_cfg):
        assert decode_cutana_raw_images([], asinh_cfg) == []


class TestBatchHomogeneity:
    def test_mixed_band_counts_raise(self, asinh_cfg):
        """A mis-wired caller passing mixed band counts fails loudly rather than
        combining with the wrong matrix / racing on the shared fb_cfg."""
        raws = [
            np.zeros((24, 24, 4), dtype=np.float32),
            np.zeros((24, 24, 3), dtype=np.float32),
        ]
        with pytest.raises(ValueError, match="homogeneous batch"):
            decode_cutana_raw_images(raws, asinh_cfg)
