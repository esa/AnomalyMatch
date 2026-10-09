#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for the persistent labeled-cache path derivation (issue #379)."""

import os
import threading
import time
from unittest.mock import patch

import pytest
from filelock import FileLock

from anomaly_match_ui.utils.backend_interface import (
    _acquire_or_bail,
    _derive_labeled_cache_path,
)


class TestDeriveLabeledCachePath:
    """Tests for _derive_labeled_cache_path."""

    def test_none_label_file_falls_back_to_output_dir(self, tmp_path):
        output_dir = str(tmp_path / "session")
        assert _derive_labeled_cache_path(None, output_dir) == os.path.join(
            output_dir, "labeled_cache"
        )

    def test_empty_label_file_falls_back(self, tmp_path):
        output_dir = str(tmp_path / "session")
        assert _derive_labeled_cache_path("", output_dir) == os.path.join(
            output_dir, "labeled_cache"
        )

    def test_writable_parent_uses_sibling_cache(self, tmp_path):
        label_file = tmp_path / "labels.csv"
        label_file.write_text("id,label\n")
        output_dir = str(tmp_path / "session")

        result = _derive_labeled_cache_path(str(label_file), output_dir)
        assert result == str(tmp_path / "labels_cache")

    def test_stem_preserved_with_multiple_dots(self, tmp_path):
        """Only the final extension is stripped — intermediate dots stay."""
        label_file = tmp_path / "run_1.final.csv"
        label_file.write_text("id,label\n")
        output_dir = str(tmp_path / "session")

        result = _derive_labeled_cache_path(str(label_file), output_dir)
        assert result == str(tmp_path / "run_1.final_cache")

    def test_unwritable_parent_falls_back_to_output_dir(self, tmp_path):
        """Simulate a read-only label-file parent by patching mkstemp."""
        label_file = tmp_path / "labels.csv"
        label_file.write_text("id,label\n")
        output_dir = str(tmp_path / "session")
        fallback = os.path.join(output_dir, "labeled_cache")

        with patch(
            "anomaly_match_ui.utils.backend_interface.tempfile.mkstemp",
            side_effect=PermissionError("read-only filesystem"),
        ):
            result = _derive_labeled_cache_path(str(label_file), output_dir)

        assert result == fallback

    def test_relative_label_file_is_absolutised(self, tmp_path, monkeypatch):
        """Relative paths get resolved so the cache path is unambiguous."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "labels.csv").write_text("id,label\n")

        result = _derive_labeled_cache_path("labels.csv", str(tmp_path / "session"))
        assert os.path.isabs(result)
        assert result == str(tmp_path / "labels_cache")


class TestCrossSessionCacheReuse:
    """Build once with one output_dir, ask again with a different one — path is stable."""

    def test_same_label_file_yields_same_cache_path(self, tmp_path):
        label_file = tmp_path / "shared_labels.csv"
        label_file.write_text("id,label\n")

        path_a = _derive_labeled_cache_path(str(label_file), str(tmp_path / "session_a"))
        path_b = _derive_labeled_cache_path(str(label_file), str(tmp_path / "session_b"))

        assert path_a == path_b == str(tmp_path / "shared_labels_cache")


class TestAcquireOrBail:
    """Tests for the stale-aware lock-acquisition helper."""

    def test_acquires_when_uncontended(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        with _acquire_or_bail(cache_dir, stale_check=None) as lock:
            assert lock is not None
            assert lock.is_locked

    def test_bails_when_stale_before_acquire(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        # Hold the underlying lock from a peer so our helper has to wait.
        peer = FileLock(f"{cache_dir}.lock")
        peer.acquire()
        try:
            # stale_check returns True on the first poll, so the helper
            # yields None without timing out.
            with _acquire_or_bail(cache_dir, stale_check=lambda: True) as lock:
                assert lock is None
        finally:
            peer.release()

    def test_releases_lock_on_exit(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        with _acquire_or_bail(cache_dir, stale_check=None) as lock:
            assert lock.is_locked
        # After the context exits, a fresh acquire must succeed.
        probe = FileLock(f"{cache_dir}.lock", timeout=1)
        probe.acquire()
        probe.release()

    def test_bails_mid_wait_when_becoming_stale(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        peer = FileLock(f"{cache_dir}.lock")
        peer.acquire()
        stale = threading.Event()
        try:
            start = time.monotonic()
            # Flip the stale flag after a brief delay to simulate a
            # newer job overtaking this one while the peer is still
            # holding the lock.
            threading.Timer(0.5, stale.set).start()
            with _acquire_or_bail(cache_dir, stale_check=stale.is_set) as lock:
                elapsed = time.monotonic() - start
                assert lock is None
                # Helper polls every ~1 s; it should notice staleness
                # well under the full 60 s timeout.
                assert elapsed < 5.0
        finally:
            peer.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
