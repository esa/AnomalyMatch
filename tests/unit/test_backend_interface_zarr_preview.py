#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for ``BackendInterface._load_images_from_zarr`` gallery preview loading."""

from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np
import pandas as pd
import zarr

from anomaly_match_ui.utils import backend_interface
from anomaly_match_ui.utils.backend_interface import _load_images_from_zarr


def _make_zarr(path, n_images: int) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("images", shape=(n_images, 4, 4, 3), chunks=(1, 4, 4, 3), dtype=np.uint8)


def test_loads_images_by_synthetic_filename(tmp_path):
    _make_zarr(tmp_path / "store.zarr", n_images=3)

    results = _load_images_from_zarr(str(tmp_path), {"store__image_000000", "store__image_000002"})

    assert set(results) == {"store__image_000000", "store__image_000002"}


def test_loads_images_by_real_metadata_filename(tmp_path):
    """A store with a metadata sidecar is looked up by its real filenames
    (e.g. "4231499475.jpeg"), not the synthetic prefix__image_NNNNNN
    fallback — this is the common case for any dataset with metadata,
    including the shared zarr test fixture.
    """
    _make_zarr(tmp_path / "store.zarr", n_images=3)
    pd.DataFrame({"filename": ["a.jpeg", "b.jpeg", "c.jpeg"]}).to_parquet(
        tmp_path / "store_metadata.parquet", index=False
    )

    results = _load_images_from_zarr(str(tmp_path), {"a.jpeg", "c.jpeg", "nonexistent.jpeg"})

    assert set(results) == {"a.jpeg", "c.jpeg"}


def test_stale_filename_is_skipped_not_crashed(tmp_path):
    """A DB row referring to a filename the store no longer produces (e.g.
    the store shrank and was rebuilt) must be dropped, not crash.
    """
    _make_zarr(tmp_path / "store.zarr", n_images=2)

    results = _load_images_from_zarr(str(tmp_path), {"store__image_000000", "store__image_000099"})

    assert set(results) == {"store__image_000000"}


def test_search_dir_is_the_zarr_store_itself(tmp_path):
    """Selecting the store directly resolve images, matching detection's handling of the same case."""
    store = tmp_path / "test_images.zarr"
    _make_zarr(store, n_images=2)

    results = _load_images_from_zarr(str(store), {"test_images__image_000001"})

    assert set(results) == {"test_images__image_000001"}


def test_batch_folder_with_nested_images_zarr(tmp_path):
    batch_dir = tmp_path / "batch_000"
    batch_dir.mkdir()
    _make_zarr(batch_dir / "images.zarr", n_images=2)

    results = _load_images_from_zarr(str(tmp_path), {"batch_000__image_000000"})

    assert set(results) == {"batch_000__image_000000"}


def test_metadata_is_read_once_across_repeated_calls(tmp_path):
    """The per-store metadata parquet is not read on every page but reads once and is then cached."""
    _make_zarr(tmp_path / "store.zarr", n_images=3)
    pd.DataFrame({"filename": ["a.jpeg", "b.jpeg", "c.jpeg"]}).to_parquet(
        tmp_path / "store_metadata.parquet", index=False
    )

    real_derive = backend_interface._derive_zarr_filenames
    with patch.object(
        backend_interface, "_derive_zarr_filenames", side_effect=real_derive
    ) as mock_derive:
        for fn in ("a.jpeg", "b.jpeg", "c.jpeg"):
            results = _load_images_from_zarr(str(tmp_path), {fn})
            assert set(results) == {fn}

    mock_derive.assert_called_once()


def test_transient_derive_failure_is_retried_not_cached_as_empty(tmp_path):
    """A store that fails to yield filenames once (e.g. a transient NFS read
    error — ``derive_filenames`` returns ``[]`` for that) must not be marked
    "known" forever: the missing wanted filename triggers one rebuild within
    the same call, so the first call already recovers instead of serving a
    permanently empty result for the kernel's lifetime.
    """
    _make_zarr(tmp_path / "store.zarr", n_images=2)

    real_derive = backend_interface._derive_zarr_filenames
    calls = {"n": 0}

    def flaky_derive(zarr_path):
        calls["n"] += 1
        return [] if calls["n"] == 1 else real_derive(zarr_path)

    with patch.object(backend_interface, "_derive_zarr_filenames", side_effect=flaky_derive):
        results = _load_images_from_zarr(str(tmp_path), {"store__image_000000"})

    assert set(results) == {"store__image_000000"}
    assert calls["n"] == 2


def test_fallback_name_poisoning_is_retried_within_one_call(tmp_path):
    """A metadata parquet that fails to read makes ``derive_filenames`` fall
    back to non-empty generated names, so the empty-list retry in
    ``_get_zarr_name_index`` can't detect the failure and marks the store
    known with the wrong names. The real filename then has to resolve within
    the same call by rebuilding the index once, instead of staying poisoned
    for the kernel's lifetime (reproduces the #612 review finding).
    """
    _make_zarr(tmp_path / "store.zarr", n_images=2)
    pd.DataFrame({"filename": ["a.jpeg", "b.jpeg"]}).to_parquet(
        tmp_path / "store_metadata.parquet", index=False
    )

    real_derive = backend_interface._derive_zarr_filenames
    calls = {"n": 0}

    def flaky_derive(zarr_path):
        calls["n"] += 1
        if calls["n"] == 1:
            return ["store__image_000000", "store__image_000001"]
        return real_derive(zarr_path)

    with patch.object(backend_interface, "_derive_zarr_filenames", side_effect=flaky_derive):
        results = _load_images_from_zarr(str(tmp_path), {"a.jpeg"})

    assert set(results) == {"a.jpeg"}
    assert calls["n"] == 2


def test_permanently_missing_name_rebuilds_only_once(tmp_path):
    """A stale filename no store produces (e.g. a DB row from a deleted batch)
    triggers one rebuild, not one per gallery page.
    """
    _make_zarr(tmp_path / "store.zarr", n_images=2)

    real_derive = backend_interface._derive_zarr_filenames
    with patch.object(
        backend_interface, "_derive_zarr_filenames", side_effect=real_derive
    ) as mock_derive:
        for _ in range(5):
            results = _load_images_from_zarr(
                str(tmp_path), {"store__image_000000", "deleted__image_000000"}
            )
            assert set(results) == {"store__image_000000"}

    # Initial build plus the single rebuild for the missing name.
    assert mock_derive.call_count == 2


def test_missing_name_is_retried_when_a_new_store_appears(tmp_path):
    """A name given up on is looked up again once a new store is written, as
    chunked prediction adds batch stores mid-run.
    """
    _make_zarr(tmp_path / "store.zarr", n_images=1)
    wanted = {"later__image_000000"}

    assert _load_images_from_zarr(str(tmp_path), wanted) == {}

    _make_zarr(tmp_path / "later.zarr", n_images=1)

    assert set(_load_images_from_zarr(str(tmp_path), wanted)) == wanted


def test_name_given_up_after_failed_rebuild_is_retried_after_expiry(tmp_path):
    """A parquet read that fails during the rebuild as well gives the real name
    up, with a warning, but only until the retry window expires; one transient
    NFS error must not blank the image for the kernel's lifetime.
    """
    _make_zarr(tmp_path / "store.zarr", n_images=2)
    pd.DataFrame({"filename": ["a.jpeg", "b.jpeg"]}).to_parquet(
        tmp_path / "store_metadata.parquet", index=False
    )

    real_derive = backend_interface._derive_zarr_filenames
    calls = {"n": 0}

    def flaky_derive(zarr_path):
        calls["n"] += 1
        if calls["n"] <= 2:
            return ["store__image_000000", "store__image_000001"]
        return real_derive(zarr_path)

    clock = {"now": 1000.0}
    with (
        patch.object(backend_interface, "_derive_zarr_filenames", side_effect=flaky_derive),
        patch.object(backend_interface.time, "monotonic", side_effect=lambda: clock["now"]),
        patch.object(backend_interface.logger, "warning") as mock_warning,
    ):
        assert _load_images_from_zarr(str(tmp_path), {"a.jpeg"}) == {}
        assert mock_warning.call_count == 1

        # Inside the retry window: no further metadata reads.
        assert _load_images_from_zarr(str(tmp_path), {"a.jpeg"}) == {}
        assert calls["n"] == 2

        clock["now"] += backend_interface._ZARR_NAME_RETRY_SECONDS + 1
        assert set(_load_images_from_zarr(str(tmp_path), {"a.jpeg"})) == {"a.jpeg"}


def test_concurrent_loads_while_stores_are_added(tmp_path):
    """Gallery prefetch threads load while chunked prediction writes new batch
    stores; the shared name-index caches must not raise or lose names.
    """
    _make_zarr(tmp_path / "store_000.zarr", n_images=2)
    errors: list[BaseException] = []

    def load_repeatedly():
        try:
            for _ in range(20):
                _load_images_from_zarr(
                    str(tmp_path), {"store_000__image_000000", "missing__image_000000"}
                )
        except BaseException as exc:  # re-raised in the main thread below
            errors.append(exc)

    threads = [threading.Thread(target=load_repeatedly) for _ in range(4)]
    for thread in threads:
        thread.start()
    for i in range(1, 6):
        _make_zarr(tmp_path / f"store_{i:03d}.zarr", n_images=1)
    for thread in threads:
        thread.join()

    assert not errors, errors
    wanted = {f"store_{i:03d}__image_000000" for i in range(6)}
    assert set(_load_images_from_zarr(str(tmp_path), wanted)) == wanted


def test_new_store_does_not_rebuild_the_index_for_given_up_names(tmp_path):
    """Chunked prediction writes a store per batch; each new store must only
    have its own metadata read, not trigger a full rebuild for stale names.
    """
    _make_zarr(tmp_path / "store_000.zarr", n_images=1)
    wanted = {"store_000__image_000000", "deleted__image_000000"}
    real_derive = backend_interface._derive_zarr_filenames

    with patch.object(
        backend_interface, "_derive_zarr_filenames", side_effect=real_derive
    ) as mock_derive:
        _load_images_from_zarr(str(tmp_path), wanted)  # build + one rebuild
        assert mock_derive.call_count == 2
        for i in range(1, 4):
            _make_zarr(tmp_path / f"store_{i:03d}.zarr", n_images=1)
            _load_images_from_zarr(str(tmp_path), wanted)

    # One read per new store, no rebuild of the older ones.
    assert mock_derive.call_count == 2 + 3
