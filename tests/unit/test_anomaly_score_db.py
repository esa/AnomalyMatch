#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for AnomalyScoreDB."""

from __future__ import annotations

import random
import sqlite3
import threading

import numpy as np
import pytest
from dotmap import DotMap

from anomaly_match.prediction.anomaly_score_db import (
    SCHEMA_VERSION,
    AnomalyScoreDB,
    SchemaVersionError,
    build_compat_metadata,
)


def _make_cfg(model_path, **overrides) -> DotMap:
    """Build a minimal cfg with the keys ``build_compat_metadata`` reads."""
    cfg = DotMap(_dynamic=False)
    cfg.model_path = str(model_path)
    cfg.net = overrides.get("net", "efficientnet-lite0")
    cfg.normalisation = DotMap(_dynamic=False)
    cfg.normalisation.image_size = overrides.get("image_size", [64, 64])
    cfg.normalisation.normalisation_method = overrides.get("normalisation_method", "ZSCALE")
    cfg.normalisation.n_output_channels = overrides.get("n_output_channels", 3)
    return cfg


@pytest.fixture()
def db(tmp_path):
    """Create a fresh AnomalyScoreDB in a temp directory."""
    db = AnomalyScoreDB(tmp_path / "predictions.db")
    yield db
    db.close()


# ── CRUD basics ──────────────────────────────────────────────────────


class TestStoreAndQuery:
    """Basic insert and query operations."""

    def test_store_and_get_count(self, db):
        db.store_results([("img_001.jpg", 0.95), ("img_002.jpg", 0.10)])
        assert db.get_count() == 2

    def test_upsert_replaces_score(self, db):
        db.store_results([("img_001.jpg", 0.5)])
        db.store_results([("img_001.jpg", 0.9)])
        assert db.get_count() == 1
        results = db.get_results(sort_by="score_desc", limit=1)
        # fp32(0.9) ≈ 0.8999999761581421
        assert results[0]["score"] == pytest.approx(0.9, abs=0.01)

    def test_get_results_returns_dicts(self, db):
        db.store_results([("a.jpg", 0.7)])
        rows = db.get_results()
        assert len(rows) == 1
        row = rows[0]
        assert set(row.keys()) == {"filename", "score"}
        assert row["filename"] == "a.jpg"


# ── Sorting ──────────────────────────────────────────────────────────


class TestSorting:
    """Verify all sort modes."""

    @pytest.fixture(autouse=True)
    def _seed_data(self, db):
        db.store_results(
            [
                ("low.jpg", 0.1),
                ("mid.jpg", 0.5),
                ("high.jpg", 0.9),
            ]
        )

    def test_score_desc(self, db):
        rows = db.get_results(sort_by="score_desc")
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_score_asc(self, db):
        rows = db.get_results(sort_by="score_asc")
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores)

    def test_updated_desc(self, db):
        """'Most recent' sort: re-upserted row moves to the top."""
        # After the fixture seeds [low, mid, high], touching 'low.jpg'
        # bumps it to the newest rowid → first in updated_desc order.
        db.store_results([("low.jpg", 0.1)])
        rows = db.get_results(sort_by="updated_desc")
        assert rows[0]["filename"] == "low.jpg"

    def test_random_returns_all(self, db):
        rows = db.get_results(sort_by="random")
        assert len(rows) == 3

    def test_score_mean_dist(self, db):
        """Closest-to-mean sort: mid (0.5) is nearest to mean(0.1, 0.5, 0.9) = 0.5."""
        rows = db.get_results(sort_by="score_mean_dist")
        filenames = [r["filename"] for r in rows]
        assert filenames[0] == "mid.jpg"

    def test_score_median_dist(self, db):
        """Closest-to-median sort: mid (0.5) is nearest to median = 0.5."""
        rows = db.get_results(sort_by="score_median_dist")
        filenames = [r["filename"] for r in rows]
        assert filenames[0] == "mid.jpg"

    def test_score_mean_dist_asymmetric(self, db):
        """Mean-distance sort with skewed data where mean != median."""
        db.store_results([("extreme.jpg", 10.0)])
        # Scores: 0.1, 0.5, 0.9, 10.0 → mean ≈ 2.875
        rows = db.get_results(sort_by="score_mean_dist")
        filenames = [r["filename"] for r in rows]
        # 0.9 is closest to 2.875 (dist 1.975), then 0.5 (2.375), then 0.1 (2.775), then 10.0 (7.125)
        assert filenames[0] == "high.jpg"

    def test_score_median_dist_even_count(self, db):
        """Median-distance with even number of rows averages two central elements."""
        db.store_results([("extra.jpg", 0.6)])
        # Scores sorted: 0.1, 0.5, 0.6, 0.9 → median = (0.5 + 0.6) / 2 = 0.55
        rows = db.get_results(sort_by="score_median_dist")
        # 0.5 is closest (dist 0.05), then 0.6 (dist 0.05), then 0.1 (0.45), then 0.9 (0.35)
        assert rows[0]["filename"] in ("mid.jpg", "extra.jpg")  # both at dist 0.05

    def test_invalid_sort_raises(self, db):
        with pytest.raises(ValueError, match="Unknown sort_by"):
            db.get_results(sort_by="invalid")


# ── Pagination ───────────────────────────────────────────────────────


class TestPagination:
    """Limit and offset queries."""

    @pytest.fixture(autouse=True)
    def _seed_data(self, db):
        db.store_results([(f"img_{i:03d}.jpg", i / 100) for i in range(50)])

    def test_limit(self, db):
        rows = db.get_results(sort_by="score_desc", limit=10)
        assert len(rows) == 10

    def test_offset(self, db):
        page1 = db.get_results(sort_by="score_desc", limit=10, offset=0)
        page2 = db.get_results(sort_by="score_desc", limit=10, offset=10)
        fns_1 = {r["filename"] for r in page1}
        fns_2 = {r["filename"] for r in page2}
        assert fns_1.isdisjoint(fns_2)

    def test_offset_beyond_end(self, db):
        rows = db.get_results(sort_by="score_desc", limit=10, offset=1000)
        assert rows == []


# ── fp32 precision ───────────────────────────────────────────────────


class TestFp32Precision:
    """Scores should be rounded to fp32 before storage."""

    def test_fp32_roundtrip(self, db):
        original = 0.123456789
        expected = float(np.float32(original))
        db.store_results([("x.jpg", original)])
        rows = db.get_results()
        assert rows[0]["score"] == pytest.approx(expected, abs=1e-9)

    def test_scores_closer_than_fp16_spacing_stay_distinct(self, db):
        """Guards the fp16 -> fp32 switch: near-ties must not collapse.

        fp16 steps by ~4.9e-4 above 0.5, which merged adjacent ranks in the
        top-scoring band; fp32 resolves them.
        """
        a, b = 0.90000, 0.90005
        db.store_results([("a.jpg", a), ("b.jpg", b)])
        rows = db.get_results(sort_by="score_desc")
        assert [r["filename"] for r in rows] == ["b.jpg", "a.jpg"]
        assert rows[0]["score"] != rows[1]["score"]
        assert float(np.float16(a)) == float(np.float16(b))

    def test_raw_integer_order_matches_float_order(self, db):
        """The whole encoding rests on uint32 sort == float sort.

        ``score_desc`` sorts the undecoded INTEGER column so it can use
        ``idx_results_score``; that is only correct because non-negative
        fp32 bit patterns are monotonic as unsigned ints.
        """
        rng = random.Random(20260901)
        scores = [rng.random() for _ in range(500)] + [0.0, 1.0]
        db.store_results([(f"s_{i:04d}.jpg", v) for i, v in enumerate(scores)])

        got = [r["score"] for r in db.get_results(sort_by="score_desc", limit=len(scores))]
        assert got == sorted((float(np.float32(v)) for v in scores), reverse=True)

    def test_fp32_extreme_values(self, db):
        db.store_results([("zero.jpg", 0.0), ("one.jpg", 1.0)])
        rows = db.get_results(sort_by="score_asc")
        assert rows[0]["score"] == 0.0
        assert rows[1]["score"] == 1.0

    def test_nan_coerced_to_zero(self, db):
        """NaN scores must not sort to the top; coerce to 0.0 on encode."""
        db.store_results([("nan.jpg", float("nan")), ("real.jpg", 0.5)])
        rows = db.get_results(sort_by="score_desc")
        assert rows[0]["filename"] == "real.jpg"
        assert rows[1]["filename"] == "nan.jpg"
        assert rows[1]["score"] == 0.0


# ── Histogram ────────────────────────────────────────────────────────


class TestHistogram:
    """Score histogram via SQL."""

    def test_histogram_shape(self, db):
        db.store_results([(f"img_{i}.jpg", i / 100) for i in range(100)])
        counts, edges = db.get_score_histogram(bins=10)
        assert len(counts) == 10
        assert len(edges) == 11
        assert counts.sum() == 100
        # Pin the per-bin counts, so the fp32() UDF decoding inside the SQL is
        # covered rather than just the row total.  Not ten per bin: 0.7 and 0.9
        # are not representable in fp32 and round *down* (0.69999999,
        # 0.89999998), so each lands one bin low.  That is the stored precision
        # showing through, and matches ``np.histogram`` over the same values.
        assert list(counts) == [10, 10, 10, 10, 10, 10, 11, 9, 11, 9]

    def test_histogram_bins_cover_range(self, db):
        db.store_results([("lo.jpg", 0.0), ("hi.jpg", 1.0)])
        counts, edges = db.get_score_histogram(bins=5)
        assert edges[0] == 0.0
        assert edges[-1] == 1.0
        assert counts.sum() == 2

    def test_histogram_empty_db(self, db):
        counts, edges = db.get_score_histogram(bins=10)
        assert counts.sum() == 0


# ── Resume ───────────────────────────────────────────────────────────


class TestResume:
    """Resume support via is_processed / get_unprocessed."""

    def test_is_processed(self, db):
        db.store_results([("done.jpg", 0.5)])
        assert db.is_processed("done.jpg") is True
        assert db.is_processed("pending.jpg") is False

    def test_get_unprocessed(self, db):
        db.store_results([("a.jpg", 0.1), ("b.jpg", 0.2)])
        unprocessed = db.get_unprocessed(["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
        assert set(unprocessed) == {"c.jpg", "d.jpg"}

    def test_get_unprocessed_all_done(self, db):
        db.store_results([("x.jpg", 0.5)])
        assert db.get_unprocessed(["x.jpg"]) == []

    def test_get_unprocessed_preserves_order_and_duplicates(self, db):
        db.store_results([("b.jpg", 0.2)])
        assert db.get_unprocessed(["c.jpg", "b.jpg", "a.jpg", "c.jpg"]) == [
            "c.jpg",
            "a.jpg",
            "c.jpg",
        ]

    def test_get_unprocessed_spans_multiple_query_chunks(self, db):
        """A batch larger than the internal chunk size resolves correctly."""
        names = [f"img_{i}.jpg" for i in range(2000)]
        db.store_results([(names[i], 0.5) for i in range(0, 2000, 2)])  # evens processed
        unprocessed = db.get_unprocessed(names)
        assert unprocessed == [names[i] for i in range(1, 2000, 2)]  # odds remain

    def test_get_unprocessed_takes_no_write_lock(self, db):
        """The resume check must not open a write transaction (the actual fix).

        The old temp-table implementation left the connection in a write
        transaction; this is the oracle that fails on it and passes on the
        SELECT-only version — the filtering-semantics tests above pass on both.
        """
        db.store_results([("a.jpg", 0.1)])
        db.get_unprocessed(["a.jpg", "b.jpg"])
        assert db._conn.in_transaction is False

    def test_get_processed_filenames(self, db):
        db.store_results([("a.jpg", 0.1), ("b.jpg", 0.2)])
        assert db.get_processed_filenames() == {"a.jpg", "b.jpg"}


# ── Compatibility validation ─────────────────────────────────────────


class TestCompatibility:
    """Config compatibility validation."""

    @pytest.fixture()
    def model_a(self, tmp_path):
        path = tmp_path / "model_a.safetensors"
        path.write_bytes(b"model contents A")
        return path

    @pytest.fixture()
    def model_b(self, tmp_path):
        path = tmp_path / "model_b.safetensors"
        path.write_bytes(b"model contents B")
        return path

    @pytest.fixture(autouse=True)
    def _seed_metadata(self, db, model_a):
        db.set_metadata_batch(build_compat_metadata(_make_cfg(model_a)))

    def test_compatible(self, db, model_a):
        ok, msg = db.validate_compatibility(_make_cfg(model_a))
        assert ok is True
        assert msg == ""

    def test_incompatible_model_contents(self, db, model_b):
        ok, msg = db.validate_compatibility(_make_cfg(model_b))
        assert ok is False
        assert "Model checkpoint contents" in msg

    def test_model_path_not_checked(self, db, model_a, tmp_path):
        """Different model_path strings must not trigger a mismatch —
        the content hash is the compatibility key, the path is just
        there for debugging."""
        twin = tmp_path / "elsewhere" / "same.safetensors"
        twin.parent.mkdir()
        twin.write_bytes(model_a.read_bytes())
        ok, msg = db.validate_compatibility(_make_cfg(twin))
        assert ok is True
        assert msg == ""

    def test_incompatible_image_size(self, db, model_a):
        ok, msg = db.validate_compatibility(_make_cfg(model_a, image_size=[128, 128]))
        assert ok is False
        assert "Image size" in msg

    def test_fresh_db_always_compatible(self, tmp_path, model_a):
        fresh = AnomalyScoreDB(tmp_path / "fresh.db")
        ok, msg = fresh.validate_compatibility(_make_cfg(model_a))
        assert ok is True
        fresh.close()


# ── Metadata ─────────────────────────────────────────────────────────


class TestMetadata:
    """Metadata storage and retrieval."""

    def test_set_and_get(self, db):
        db.set_metadata("model_path", "/path/to/model.safetensors")
        meta = db.get_metadata()
        assert meta["model_path"] == "/path/to/model.safetensors"

    def test_set_metadata_batch(self, db):
        db.set_metadata_batch({"a": 1, "b": [2, 3], "c": {"nested": True}})
        meta = db.get_metadata()
        assert meta["a"] == 1
        assert meta["b"] == [2, 3]
        assert meta["c"] == {"nested": True}

    def test_metadata_overwrite(self, db):
        db.set_metadata("key", "old")
        db.set_metadata("key", "new")
        assert db.get_metadata()["key"] == "new"

    def test_schema_version_recorded(self, db):
        meta = db.get_metadata()
        assert meta["schema_version"] == SCHEMA_VERSION


# ── Long-run resilience ──────────────────────────────────────────────


class TestHeartbeat:
    """Heartbeat is advanced on every store_results commit."""

    def test_no_heartbeat_on_fresh_db(self, db):
        assert db.get_heartbeat() is None

    def test_heartbeat_updated_on_store(self, db):
        import time

        before = int(time.time())
        db.store_results([("a.jpg", 0.5)])
        hb = db.get_heartbeat()
        assert hb is not None
        assert hb >= before

    def test_heartbeat_advances_across_batches(self, db):
        import time

        db.store_results([("a.jpg", 0.1)])
        first = db.get_heartbeat()
        time.sleep(1.1)  # wall-clock tick so unix-seconds differ
        db.store_results([("b.jpg", 0.2)])
        second = db.get_heartbeat()
        assert second > first


class TestCheckpointing:
    """wal_checkpoint(TRUNCATE) runs every N commits and on close."""

    def test_wal_truncated_every_n_commits(self, tmp_path):
        """After checkpoint_every_n_commits batches, .wal is empty."""
        path = tmp_path / "ckpt.db"
        db = AnomalyScoreDB(path, checkpoint_every_n_commits=3)
        wal_path = path.with_suffix(path.suffix + "-wal")
        try:
            # Two commits — below threshold, .wal still holds pages.
            db.store_results([("a.jpg", 0.1)])
            db.store_results([("b.jpg", 0.2)])
            assert wal_path.exists()
            wal_before = wal_path.stat().st_size
            assert wal_before > 0

            # Third commit triggers TRUNCATE.
            db.store_results([("c.jpg", 0.3)])
            assert wal_path.stat().st_size == 0
        finally:
            db.close()

    def test_wal_absent_after_clean_close(self, tmp_path):
        """close() leaves no .wal behind, so a reopen sees no stale WAL."""
        path = tmp_path / "close.db"
        db = AnomalyScoreDB(path)
        db.store_results([("a.jpg", 0.1)])
        db.close()

        wal_path = path.with_suffix(path.suffix + "-wal")
        # SQLite may keep a zero-byte .wal file; what matters is that
        # nothing uncommitted survived the close.
        if wal_path.exists():
            assert wal_path.stat().st_size == 0


class TestSynchronousFull:
    """Optional paranoid mode activates synchronous=FULL."""

    def test_default_is_normal(self, db):
        row = db._conn.execute("PRAGMA synchronous").fetchone()
        # PRAGMA synchronous returns 1 for NORMAL, 2 for FULL.
        assert row[0] == 1

    def test_paranoid_mode_sets_full(self, tmp_path):
        path = tmp_path / "full.db"
        db = AnomalyScoreDB(path, synchronous_full=True)
        try:
            row = db._conn.execute("PRAGMA synchronous").fetchone()
            assert row[0] == 2
        finally:
            db.close()


class TestImmediateTransactions:
    """BEGIN IMMEDIATE + busy_timeout make two writers well-behaved."""

    def test_isolation_level_is_immediate(self, db):
        assert db._conn.isolation_level == "IMMEDIATE"

    def test_two_writers_dont_corrupt(self, tmp_path):
        """Two AnomalyScoreDB connections writing to the same file both
        complete successfully, with no SQLITE_BUSY bubbling up (busy_timeout
        covers contention) and all rows present afterwards.

        This is a forward-looking smoke test for #322: single-writer is
        the current invariant, but a future coordinator or accidental
        double-open must at least not corrupt the DB.
        """
        path = tmp_path / "concurrent.db"
        # Pre-create the DB so both writer threads skip the schema-init
        # path; the test targets concurrent writes, not concurrent opens.
        AnomalyScoreDB(path).close()

        errors: list[tuple[str, BaseException]] = []
        writers = ("A", "B")
        n_per_writer = 100

        def writer(label: str):
            try:
                conn = AnomalyScoreDB(path)
                try:
                    for i in range(n_per_writer):
                        conn.store_results([(f"{label}_{i:03d}.jpg", i / 1000.0)])
                finally:
                    conn.close()
            except BaseException as exc:  # noqa: BLE001 — re-raised via errors
                errors.append((label, exc))

        threads = [threading.Thread(target=writer, args=(w,)) for w in writers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writers raised: {errors}"
        with AnomalyScoreDB(path) as verifier:
            assert verifier.get_count() == n_per_writer * len(writers)
            all_filenames = verifier.get_processed_filenames()
            for w in writers:
                for i in range(n_per_writer):
                    assert f"{w}_{i:03d}.jpg" in all_filenames


# ── Schema version gate ──────────────────────────────────────────────


class TestSchemaVersion:
    """Reject DBs written by older schema versions."""

    def test_old_schema_rejected(self, tmp_path):
        """A DB without a run_metadata table is treated as legacy."""
        path = tmp_path / "legacy.db"
        con = sqlite3.connect(str(path))
        con.execute("CREATE TABLE results (id INTEGER PRIMARY KEY, filename TEXT, score REAL)")
        con.commit()
        con.close()

        with pytest.raises(SchemaVersionError, match="older schema"):
            AnomalyScoreDB(path)

    @pytest.mark.parametrize("stored_version", [2, 3, 99])
    def test_mismatched_schema_version_rejected(self, tmp_path, stored_version):
        """Any schema_version but the current one is refused on open.

        v3 matters most: it holds fp16 bit patterns in the same INTEGER
        column as v4, so nothing about its shape distinguishes it and only
        this check stops it from decoding to garbage scores.
        """
        path = tmp_path / f"v{stored_version}.db"
        con = sqlite3.connect(str(path))
        con.executescript(
            "CREATE TABLE run_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE results (id INTEGER PRIMARY KEY, filename TEXT, score INTEGER);"
        )
        con.execute(
            "INSERT INTO run_metadata VALUES (?, ?)", ("schema_version", str(stored_version))
        )
        con.commit()
        con.close()

        with pytest.raises(SchemaVersionError, match=f"schema_version={stored_version}"):
            AnomalyScoreDB(path)

    def test_reopen_own_db_succeeds(self, tmp_path):
        """A DB written by this version must reopen cleanly."""
        path = tmp_path / "roundtrip.db"
        db1 = AnomalyScoreDB(path)
        db1.store_results([("a.jpg", 0.5)])
        db1.close()

        db2 = AnomalyScoreDB(path)
        assert db2.get_count() == 1
        db2.close()


# ── Convenience methods ──────────────────────────────────────────────


class TestConvenience:
    """get_top_results, get_all_scores, get_score_range."""

    @pytest.fixture(autouse=True)
    def _seed_data(self, db):
        db.store_results(
            [
                ("lo.jpg", 0.1),
                ("mid.jpg", 0.5),
                ("hi.jpg", 0.9),
            ]
        )

    def test_get_top_results(self, db):
        top = db.get_top_results(2)
        assert len(top) == 2
        assert top[0]["score"] >= top[1]["score"]

    def test_get_all_scores(self, db):
        scores = db.get_all_scores()
        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float32
        # Pin the values, not just the shape: this is the one read path whose
        # body is not a mechanical rename ("<u4" viewed as "<f4"), so a wrong
        # view would still produce three float32s of the right length.
        assert np.array_equal(np.sort(scores), np.array([0.1, 0.5, 0.9], dtype=np.float32))

    def test_get_score_range(self, db):
        lo, hi = db.get_score_range()
        assert lo == pytest.approx(float(np.float32(0.1)), abs=1e-3)
        assert hi == pytest.approx(float(np.float32(0.9)), abs=1e-3)

    def test_get_score_range_empty_raises(self, db, tmp_path):
        empty_db = AnomalyScoreDB(tmp_path / "empty.db")
        with pytest.raises(ValueError, match="No results"):
            empty_db.get_score_range()
        empty_db.close()


# ── Context manager / lifecycle ──────────────────────────────────────


class TestLifecycle:
    """Context manager and repr."""

    def test_context_manager(self, tmp_path):
        with AnomalyScoreDB(tmp_path / "ctx.db") as db:
            db.store_results([("a.jpg", 0.5)])
            assert db.get_count() == 1

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "predictions.db"
        db = AnomalyScoreDB(nested)
        assert nested.exists()
        db.close()


# ── Concurrent access ────────────────────────────────────────────────


class TestConcurrency:
    """WAL mode allows reader + writer threads simultaneously."""

    def test_concurrent_read_write(self, db):
        """Writer inserts while a reader queries — no locking errors or data loss.

        Mirrors production: the prediction subprocess writes through its own
        connection while the UI (BackendInterface) reads through a *separate*
        connection to the same file. Sharing one sqlite connection across
        threads is not how the app runs — it races on the writer's transaction
        state — so the reader opens its own ``AnomalyScoreDB`` here.
        """
        errors = []
        n_writes = 200
        n_reads = 50

        def writer():
            try:
                for i in range(n_writes):
                    db.store_results([(f"w_{i}.jpg", i / n_writes)])
            except Exception as exc:
                errors.append(("writer", exc))

        def reader():
            try:
                with AnomalyScoreDB(db._db_path) as reader_db:
                    for _ in range(n_reads):
                        reader_db.get_results(sort_by="score_desc", limit=10)
                        reader_db.get_count()
            except Exception as exc:
                errors.append(("reader", exc))

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert errors == [], f"Concurrency errors: {errors}"
        assert db.get_count() == n_writes


# ── Scale + on-disk size ─────────────────────────────────────────────


class TestScale:
    """Query performance and storage footprint at moderate scale."""

    def test_query_10k_entries(self, db):
        """Insert 10k entries and verify sorted paginated query works."""
        batch = [(f"img_{i:06d}.jpg", np.random.random()) for i in range(10_000)]
        db.store_results(batch)
        assert db.get_count() == 10_000

        # Sorted paginated query
        page = db.get_results(sort_by="score_desc", limit=20, offset=100)
        assert len(page) == 20
        scores = [r["score"] for r in page]
        assert scores == sorted(scores, reverse=True)

        # Histogram
        counts, edges = db.get_score_histogram(bins=20)
        assert counts.sum() == 10_000

    def test_on_disk_size_regression(self, tmp_path):
        """Pin the per-row byte budget so schema regressions are visible.

        10k rows with 19-char numeric filenames (the cutana-style
        worst case on the observed 286 MB production DB, which ran
        ~170 B/row).  Budget is 90 B/row against the ~81.5 B/row this
        layout actually costs — fp32 scores took ~4 B/row of the former
        margin, so there is now ~8.5 B/row of headroom, not ~15.  Breach
        this and the whole point of the schema slim is gone.
        """
        path = tmp_path / "size.db"
        db = AnomalyScoreDB(path)
        batch = [(f"{2752727549657434000 + i:019d}", (i % 1000) / 1000.0) for i in range(10_000)]
        db.store_results(batch)
        db.close()

        size_bytes = path.stat().st_size
        assert size_bytes < 10_000 * 90, (
            f"predictions.db grew to {size_bytes} bytes "
            f"({size_bytes / 10_000:.1f} B/row) for 10k rows — schema regression? "
            "Re-baseline if intentional."
        )


# ── catalogue-origin index (#506) ─────────────────────────────────────


class TestCatalogueIndex:
    """Record/read which catalogue holds each scored source."""

    def test_round_trips_recorded_origins(self, db):
        db.store_results([("s1", 0.9), ("s2", 0.5), ("s3", 0.1)])
        db.record_catalogues({"s1": "/cat/a.parquet", "s2": "/cat/b.parquet"})

        got = db.get_catalogue_map(["s1", "s2", "s3"])
        assert got == {"s1": "/cat/a.parquet", "s2": "/cat/b.parquet"}

    def test_unrecorded_sources_are_absent(self, db):
        """A scored-but-unindexed source is omitted, so the loader full-scans it."""
        db.store_results([("s1", 0.9)])
        assert db.get_catalogue_map(["s1"]) == {}

    def test_interns_paths_so_index_is_reused(self, db):
        """The same catalogue path is interned once and its index is shared."""
        db.store_results([("s1", 0.9), ("s2", 0.5)])
        db.record_catalogues({"s1": "/cat/a.parquet", "s2": "/cat/a.parquet"})

        rows = db._conn.execute("SELECT idx, path FROM catalogues").fetchall()
        assert len(rows) == 1
        idxs = {r["catalogue_idx"] for r in db._conn.execute("SELECT catalogue_idx FROM results")}
        assert idxs == {rows[0]["idx"]}

    def test_re_recording_keeps_catalogue_index_stable(self, db):
        db.store_results([("s1", 0.9)])
        db.record_catalogues({"s1": "/cat/a.parquet"})
        first = db._conn.execute("SELECT idx FROM catalogues").fetchone()["idx"]
        db.record_catalogues({"s1": "/cat/a.parquet"})
        second = db._conn.execute("SELECT idx FROM catalogues").fetchone()["idx"]
        assert first == second

    def test_record_ignores_unknown_source(self, db):
        """Recording an id with no results row is a harmless no-op (not scored)."""
        db.store_results([("s1", 0.9)])
        db.record_catalogues({"ghost": "/cat/a.parquet"})
        assert db.get_catalogue_map(["s1", "ghost"]) == {}

    def test_rescore_clears_catalogue_idx(self, db):
        """INSERT OR REPLACE on re-score drops the origin; loader re-back-fills."""
        db.store_results([("s1", 0.9)])
        db.record_catalogues({"s1": "/cat/a.parquet"})
        db.store_results([("s1", 0.4)])
        assert db.get_catalogue_map(["s1"]) == {}

    def test_empty_inputs(self, db):
        db.record_catalogues({})  # no-op, must not raise
        assert db.get_catalogue_map([]) == {}

    def test_read_only_open_reads_without_writing(self, tmp_path):
        """A read-only open serves lookups but creates no tables (no write lock)."""
        path = tmp_path / "predictions.db"
        with AnomalyScoreDB(path) as db:
            db.store_results([("s1", 0.9)])
            db.record_catalogues({"s1": "/cat/a.parquet"})

        with AnomalyScoreDB(path, read_only=True) as ro:
            assert ro.get_catalogue_map(["s1"]) == {"s1": "/cat/a.parquet"}

    def test_read_only_open_skips_table_creation(self, tmp_path):
        """Opening a fresh path read-only must not create tables (proves no write)."""
        with AnomalyScoreDB(tmp_path / "fresh.db", read_only=True) as ro:
            tables = {
                row["name"]
                for row in ro._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        assert "results" not in tables
