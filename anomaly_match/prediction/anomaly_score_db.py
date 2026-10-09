#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""SQLite-backed prediction result database with resume support.

Single ``predictions.db`` per prediction run, storing ``(filename, score)``
pairs plus a ``run_metadata`` table used to validate that a resumed run
targets the same model / normalisation as the original run.

Schema (``schema_version=4``):

* ``results`` ``(id, filename, score)``: ``score`` is the 32-bit ``float`` bit
  pattern as an ``INTEGER`` (0..4294967295) — non-negative floats keep their
  ordering as unsigned ints, so ``ORDER BY score`` works. The rowid ``id`` gives
  insertion order (``INSERT OR REPLACE`` bumps it), so ``ORDER BY id DESC`` is
  "most recently written" with no ``updated_at`` column.
* ``run_metadata``: flat ``key → json`` store for the schema version and the
  :meth:`AnomalyScoreDB.validate_compatibility` fields.

Opening an older-schema DB raises :class:`SchemaVersionError` (pre-release
layout, not migrated).
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotmap import DotMap
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bumped to 4 for fp32 score storage: ``results.score`` now holds a uint32
# fp32 bit pattern where v3 held a uint16 fp16 one.  The column type is
# unchanged, so a v3 DB would decode to garbage rather than fail — the version
# bump is what makes that mismatch loud.  Older DBs are rejected on open
# (delete and re-run) — the project keeps no legacy DB compatibility.
SCHEMA_VERSION = 4

# Sort expressions.  ``score`` is an INTEGER fp32 bit pattern; for
# non-negative floats this matches float ordering, so ``score_desc``/
# ``score_asc`` can use the raw column (and the ``idx_results_score``
# index).  Distance-based sorts decode via the ``fp32`` UDF registered
# on the connection.
_SORT_CLAUSES: dict[str, str] = {
    "score_desc": "r.score DESC",
    "score_asc": "r.score ASC",
    "score_mean_dist": ("ABS(fp32(r.score) - (SELECT AVG(fp32(score)) FROM results))"),
    "score_median_dist": (
        "ABS(fp32(r.score) - ("
        "  SELECT AVG(fp32(score)) FROM ("
        "    SELECT score FROM results ORDER BY score"
        "    LIMIT 2 - (SELECT COUNT(*) FROM results) % 2"
        "    OFFSET (SELECT (COUNT(*) - 1) / 2 FROM results)"
        "  )"
        "))"
    ),
    # id is monotonic with insertion (INSERT OR REPLACE bumps rowid),
    # so ORDER BY id gives "most recent" for free — no extra index.
    "updated_desc": "r.id DESC",
    "updated_asc": "r.id ASC",
    "random": "RANDOM()",
}

# Fields checked by validate_compatibility.  Each maps a metadata key to a
# human-readable label used in the mismatch message.  ``model_sha256`` is
# a content hash of the checkpoint file; ``model_path`` is stored in
# ``run_metadata`` for debugging but intentionally not compared — the
# same file can be referenced by relative vs absolute paths and would
# otherwise falsely trigger a mismatch.
_COMPAT_KEYS: dict[str, str] = {
    "model_sha256": "Model checkpoint contents",
    "image_size": "Image size",
    "normalisation_method": "Normalisation method",
    "n_output_channels": "Number of output channels",
    "net": "Network architecture",
}

_SCHEMA_VERSION_KEY = "schema_version"
_HEARTBEAT_KEY = "last_heartbeat"

# Passive WAL autocheckpoint threshold, in pages. SQLite checkpoints on the
# *writer's own* connection whenever the WAL reaches this many frames, copying
# every frame the oldest reader no longer needs back into the main DB so those
# frames can be reused. Unlike the manual TRUNCATE below (which fires only every
# N commits and needs a zero-reader moment), this fires on WAL *size*, so it
# catches more of the brief windows when readers release their snapshots — which
# matters most when the DB is on a slow filesystem where such windows are rare.
# At 8 KiB/page, 2000 pages targets a ~16 MiB steady-state WAL.
#
# Caveat: no checkpoint — passive or TRUNCATE — can advance past a reader that
# holds an *old* snapshot open. On NFS, slow full-table UI reads pin such
# snapshots for seconds while the writer keeps appending, which is what let the
# .wal reach 11 GB (bigger WAL -> slower reads -> longer pins, a feedback loop).
# The robust fix for that is putting the DB on local disk so reads, and thus
# reader snapshots, stay short; this setting only reduces the damage.
_WAL_AUTOCHECKPOINT_PAGES = 2000

# Writer lock spin time before raising SQLITE_BUSY. Applies to normal writes,
# but is *dropped to zero* around the opportunistic TRUNCATE checkpoint in
# ``store_results`` so that checkpoint never stalls the write loop waiting on the
# live UI reader (see the checkpoint block for the measured 30 s spikes).
_BUSY_TIMEOUT_MS = 30000

# Max bound parameters per statement. SQLite's SQLITE_MAX_VARIABLE_NUMBER is 999
# by default (historically), so IN-clause reads chunk their id list under this.
_MAX_SQL_VARIABLES = 900

# Cadence (in ``store_results`` commits) for the *opportunistic* TRUNCATE
# checkpoint. TRUNCATE additionally resets the .wal file to zero bytes, but only
# when no reader holds it open — under the live UI poller it usually returns
# busy=1 and shrinks nothing. That is fine: ``_WAL_AUTOCHECKPOINT_PAGES`` already
# bounds growth, so TRUNCATE is just a best-effort shrink for the quiet moments
# between polls. Keep the cadence low so those shrink attempts are frequent.
_DEFAULT_CHECKPOINT_EVERY_N_COMMITS = 25


class SchemaVersionError(RuntimeError):
    """Raised when opening a predictions.db written by an older schema."""


def compute_model_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a model checkpoint file.

    Used by both the prediction subprocesses (when writing
    ``model_sha256`` into ``run_metadata``) and the UI resume check
    (when asking whether an existing DB is compatible with the
    currently selected model).  Content-hashing sidesteps the
    fragility of comparing ``model_path`` strings — relative vs
    absolute paths, symlinks, and cross-machine moves all collapse
    into the same hash.

    Streamed in 1 MiB chunks so a multi-GB checkpoint doesn't have to
    fit in memory.

    Args:
        path: Filesystem path to the checkpoint file.

    Returns:
        Hex digest string (64 characters).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_compat_metadata(cfg: DotMap) -> dict[str, Any]:
    """Build the run_metadata dict for a prediction config.

    Centralises the model / normalisation fields that are stored in
    ``run_metadata`` and consulted by
    :meth:`AnomalyScoreDB.validate_compatibility`.  ``model_sha256`` (not
    ``model_path``) is the identity key the compatibility check uses:
    two runs whose ``cfg.model_path`` strings differ but whose files
    have the same bytes are compatible, and two runs whose strings
    match but whose files differ are not.

    Args:
        cfg: Prediction config with ``model_path``, ``net``, and
            ``normalisation`` populated.

    Returns:
        Metadata dict suitable for :meth:`AnomalyScoreDB.set_metadata_batch`
        and :meth:`AnomalyScoreDB.validate_compatibility`.
    """
    return {
        "model_path": cfg.model_path,
        "model_sha256": compute_model_sha256(cfg.model_path),
        "net": cfg.net,
        "image_size": list(cfg.normalisation.image_size),
        "normalisation_method": str(cfg.normalisation.normalisation_method),
        "n_output_channels": cfg.normalisation.n_output_channels,
    }


def _fp32_encode(value: float) -> int:
    """Encode a float as the uint32 bit pattern of its fp32 representation.

    NaN is clamped to ``0`` so sort-by-score does not place it at the top.

    Args:
        value: Score in ``[0, 1]`` (anomaly probability).

    Returns:
        Integer in ``[0, 4294967295]`` suitable for the ``score`` column.
    """
    # No explicit np.float32() cast: struct.pack("<f") already rounds to
    # nearest-even fp32.  This runs once per stored row, so the saved
    # round-trip through numpy is worth the note (~5x on the call).
    if math.isnan(value):
        return 0
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _fp32_decode(raw: int) -> float:
    """Decode a uint32 fp32 bit pattern back to a Python float.

    ``score`` is ``INTEGER NOT NULL`` and the one caller that could see a NULL
    (``get_score_range`` on an empty table) raises before decoding, so a
    ``None`` reaching here is a broken invariant and is left to raise.

    Args:
        raw: Integer 0..4294967295.

    Returns:
        The decoded float value.
    """
    return float(struct.unpack("<f", struct.pack("<I", raw))[0])


class AnomalyScoreDB:
    """SQLite store for prediction results with resume support.

    Concurrency contract
    --------------------
    Uses WAL journal mode so the prediction subprocess (the single
    writer) and the UI process (readers) can operate concurrently
    without blocking.  Implicit transactions use ``BEGIN IMMEDIATE``
    (see :class:`sqlite3.Connection.isolation_level`) — the RESERVED
    lock is acquired at BEGIN time, before any work is done, so a
    contending writer usually waits out ``busy_timeout`` (30 s) or
    raises :class:`sqlite3.OperationalError` cleanly rather than
    racing another writer mid-transaction.  ``busy_timeout`` does *not*
    cover a lock *upgrade*, though: a connection holding SHARED that
    tries to take RESERVED while another already holds it gets
    ``SQLITE_BUSY`` immediately.  So **read paths must issue only
    ``SELECT`` — no temp tables, no DML** — otherwise ``isolation_level``
    turns a "read" into a write transaction that contends for the single
    writer slot (this bit :meth:`get_unprocessed`, which is why it is
    SELECT-only).

    **Single writer per DB is the current invariant.**  IMMEDIATE +
    ``busy_timeout`` is the primitive a future multi-writer
    coordinator (issue #322) will extend: either a queue-based single
    writer draining per-GPU workers, or per-worker DBs merged via
    ``ATTACH``.  Either path drops in without schema change.

    Args:
        db_path: Path to the SQLite database file.  Created if missing.
        checkpoint_every_n_commits: Run ``PRAGMA wal_checkpoint(TRUNCATE)``
            after this many :meth:`store_results` commits.  Bounds the
            ``.wal`` file size on multi-day runs where a reader might
            otherwise keep the WAL from being folded into the main file.
            Defaults to 1000.
        synchronous_full: If ``True``, use ``PRAGMA synchronous=FULL``
            (fsync after every commit — crash-safe but ~10% slower).
            Defaults to ``False`` (``synchronous=NORMAL``), which trusts
            the filesystem to fsync within the rollback-journal contract.
        read_only: If ``True``, open for queries only — skip the
            table-creation and schema-version writes so no write lock is
            taken. Used by the gallery's catalogue-index lookup (#506) so a
            read never contends with the live scoring writer. Defaults to
            ``False``.
        busy_timeout_ms: ``PRAGMA busy_timeout`` for this connection, in
            milliseconds — how long a statement waits on a lock held by
            another connection before raising "database is locked". Lower it
            (e.g. ``0``) for a best-effort writer that should give up
            immediately under contention rather than block. Defaults to
            :data:`_BUSY_TIMEOUT_MS`.

    Raises:
        SchemaVersionError: If *db_path* exists and was written by an
            incompatible schema version.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        db_path: str | Path,
        *,
        checkpoint_every_n_commits: int = _DEFAULT_CHECKPOINT_EVERY_N_COMMITS,
        synchronous_full: bool = False,
        read_only: bool = False,
        busy_timeout_ms: int = _BUSY_TIMEOUT_MS,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialises _safe_execute against close() to prevent use-after-free
        # of the C-level sqlite3 handle (segfault on Linux).
        self._close_lock = threading.Lock()

        self._checkpoint_every_n_commits = checkpoint_every_n_commits
        self._synchronous_full = synchronous_full
        self._commits_since_checkpoint = 0
        # read_only skips the table-creation and schema-version writes below, so
        # opening for a query (e.g. the gallery's catalogue-index lookup, #506)
        # takes no write lock and never contends with the live scoring writer.
        # busy_timeout_ms can be lowered (e.g. 0) so a best-effort writer gives
        # up immediately under contention instead of blocking.
        self._busy_timeout_ms = busy_timeout_ms

        db_existed = self._db_path.exists()

        self._conn = sqlite3.connect(
            str(self._db_path),
            # check_same_thread=False allows the UI thread to read while
            # the prediction thread writes (WAL handles the concurrency).
            check_same_thread=False,
            # IMMEDIATE upgrades implicit BEGIN to BEGIN IMMEDIATE, which
            # acquires the RESERVED lock at transaction start. Two writers
            # contending for the DB then resolve via busy_timeout instead
            # of racing mid-transaction; see the class docstring.
            isolation_level="IMMEDIATE",
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.create_function("fp32", 1, _fp32_decode, deterministic=True)

        self._configure_pragmas()
        if db_existed:
            self._assert_schema_version()
        if not read_only:
            self._create_tables()
            self._record_schema_version()

        logger.debug("AnomalyScoreDB opened at {}", self._db_path)

    # -- internal helpers ------------------------------------------------

    def _configure_pragmas(self) -> None:
        cur = self._conn.cursor()
        # Write-Ahead Logging (WAL) allows concurrent reads during writes.
        cur.execute("PRAGMA journal_mode=WAL")
        synchronous = "FULL" if self._synchronous_full else "NORMAL"
        cur.execute(f"PRAGMA synchronous={synchronous}")
        # page_size must be set before any table exists; safe to repeat.
        cur.execute("PRAGMA page_size=8192")
        cur.execute("PRAGMA cache_size=-65536")  # 64 MiB
        cur.execute("PRAGMA temp_store=MEMORY")
        # Passive autocheckpoint checkpoints on WAL size (not just the
        # every-N-commits TRUNCATE in store_results), catching more of the brief
        # reader-release windows; the manual TRUNCATE then opportunistically
        # shrinks the file to zero when a gap allows. See _WAL_AUTOCHECKPOINT_PAGES:
        # this mitigates NFS WAL bloat, but local disk is the real fix.
        cur.execute(f"PRAGMA wal_autocheckpoint={_WAL_AUTOCHECKPOINT_PAGES}")
        # A writer that finds the lock held (e.g. UI doing a long scan)
        # will spin for up to this many ms before raising SQLITE_BUSY.
        cur.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")

    def _assert_schema_version(self) -> None:
        """Refuse to open a DB written by a different schema version.

        Raises:
            SchemaVersionError: Stored ``schema_version`` differs from
                :data:`SCHEMA_VERSION`, or is missing on a non-empty DB.
        """
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='run_metadata'"
        ).fetchone()
        if row is None:
            # Pre-existing file with no run_metadata table → old schema.
            raise SchemaVersionError(
                f"{self._db_path} was written by an older schema version. "
                "Delete the file and re-run prediction."
            )

        row = cur.execute(
            "SELECT value FROM run_metadata WHERE key = ?", (_SCHEMA_VERSION_KEY,)
        ).fetchone()
        stored = json.loads(row["value"]) if row is not None else None
        if stored != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"{self._db_path} has schema_version={stored!r}, "
                f"expected {SCHEMA_VERSION}. Delete the file and re-run prediction."
            )

    def _create_tables(self) -> None:
        """Create the core tables and indices if they do not already exist."""
        cur = self._conn.cursor()
        cur.executescript(
            """
            -- Catalogue origin index (#506): which catalogue file holds each
            -- scored source, so the gallery reads only the right file instead
            -- of re-scanning every catalogue per page.  The path list is
            -- interned here and ``results.catalogue_idx`` carries only the
            -- small integer key — SQLite varint-encodes it to ~1 byte/row.
            -- Declared before ``results`` so the foreign key below names an
            -- existing parent.
            CREATE TABLE IF NOT EXISTS catalogues (
                idx  INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE
            );

            -- ``catalogue_idx`` is the small integer key into ``catalogues``
            -- (#506); NULL until the gallery back-fills it, in which case the
            -- loader falls back to a full catalogue scan for that source. The
            -- REFERENCES clause documents the relationship and lets a future
            -- ``PRAGMA foreign_keys=ON`` enforce it; SQLite leaves enforcement
            -- off per-connection by default, which is what we rely on today —
            -- the column is nullable and back-filled after the row is inserted,
            -- and ``record_catalogues`` always interns the parent path before
            -- pointing a row at it, so the reference is never dangling.
            CREATE TABLE IF NOT EXISTS results (
                id            INTEGER PRIMARY KEY,
                filename      TEXT    NOT NULL UNIQUE,
                score         INTEGER NOT NULL,
                catalogue_idx INTEGER REFERENCES catalogues(idx)
            );

            CREATE INDEX IF NOT EXISTS idx_results_score
                ON results(score);

            CREATE TABLE IF NOT EXISTS run_metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def _record_schema_version(self) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_metadata(key, value) VALUES (?, ?)",
            (_SCHEMA_VERSION_KEY, json.dumps(SCHEMA_VERSION)),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write operations (prediction thread)
    # ------------------------------------------------------------------

    def store_results(self, results: list[tuple[str, float]]) -> None:
        """Batch insert or update prediction results.

        Scores are rounded to fp32 precision before storage.  Existing
        rows with the same filename are replaced (``INSERT OR REPLACE``);
        replacement assigns a new rowid so the replaced row moves to the
        top of the ``updated_desc`` ordering.

        The commit also bumps the ``last_heartbeat`` metadata key. WAL
        growth is bounded passively by ``PRAGMA wal_autocheckpoint``; every
        ``checkpoint_every_n_commits`` commits an opportunistic
        ``PRAGMA wal_checkpoint(TRUNCATE)`` additionally tries to reset the
        .wal file to zero, which only succeeds when no reader holds it open.
        The checkpoint runs *outside* the transaction — it cannot execute
        inside one.

        Args:
            results: Sequence of ``(filename, score)`` tuples.

        Raises:
            sqlite3.DatabaseError: If the score write fails for a non-transient
                reason. A transient WAL-checkpoint lock from a concurrent reader
                is caught (the WAL stays bounded by autocheckpoint), not raised.
        """
        if not results:
            return

        rows = [(fn, _fp32_encode(score)) for fn, score in results]
        now = int(time.time())
        cur = self._conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO results(filename, score) VALUES (?, ?)",
            rows,
        )
        # Heartbeat in the same transaction as the scores — any reader
        # seeing a fresh heartbeat also sees the scores it advertises.
        cur.execute(
            "INSERT OR REPLACE INTO run_metadata(key, value) VALUES (?, ?)",
            (_HEARTBEAT_KEY, json.dumps(now)),
        )
        self._conn.commit()

        self._commits_since_checkpoint += 1
        if self._commits_since_checkpoint >= self._checkpoint_every_n_commits:
            self._commits_since_checkpoint = 0
            # Opportunistic shrink only. wal_autocheckpoint already bounds WAL
            # growth passively, so this TRUNCATE just resets the file to zero
            # when no reader holds it open. Under the live UI poller it usually
            # returns busy=1 and shrinks nothing — harmless, the WAL stays at its
            # bounded high-water mark until a quieter moment. Must never crash
            # the prediction write loop.
            #
            # Critically, drop busy_timeout to 0 for the duration: TRUNCATE
            # respects busy_timeout, so at the default 30 s it blocks the write
            # loop for the *full* 30 s every cadence interval while the gallery
            # poller holds a reader snapshot (measured: ~30 s spikes once per 25
            # commits, ~16% of prediction wall and the gap below peak img/s). At
            # 0 it gives up instantly when a reader is active and still shrinks
            # in the genuine quiet gaps. wal_autocheckpoint bounds the WAL either
            # way, so giving up costs nothing.
            try:
                cur.execute("PRAGMA busy_timeout=0")
                # Row is (busy, log_frames, checkpointed_frames); busy=1 means a
                # reader kept the WAL open so it could not be reset to zero.
                row = cur.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if row is not None and row[0]:
                    logger.debug(
                        "WAL truncate blocked by reader (busy); WAL bounded by autocheckpoint"
                    )
            except sqlite3.DatabaseError as exc:
                # A reader can block the WAL restart as either "database is
                # locked" (OperationalError) or "database table is locked"
                # (base DatabaseError). Both are transient — autocheckpoint keeps
                # the WAL bounded regardless. Anything else is a real error.
                if "locked" not in str(exc).lower():
                    raise
                logger.debug("WAL checkpoint deferred (reader active): {}", exc)
            finally:
                cur.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")

    def record_catalogues(self, source_to_catalogue: dict[str, str]) -> None:
        """Record which catalogue file each scored source came from (#506).

        Interns each distinct catalogue path into the ``catalogues`` table and
        points the matching ``results`` rows at it via ``catalogue_idx``, so a
        later gallery load reads only the catalogue that holds a source instead
        of scanning them all.  Sources without a ``results`` row (not scored)
        are silently ignored — only scored sources are ever browsed.

        Args:
            source_to_catalogue: Map from ``source_id`` (the ``results.filename``
                key) to the absolute catalogue path it was found in.  An empty
                map is a no-op; callers only record sources they actually found.
        """
        cur = self._conn.cursor()
        # Intern the distinct paths first; INSERT OR IGNORE keeps existing
        # indices stable so previously-recorded rows stay valid.  One
        # executemany rather than a Python-level loop of single executes.
        distinct_paths = {str(p) for p in source_to_catalogue.values()}
        cur.executemany(
            "INSERT OR IGNORE INTO catalogues(path) VALUES (?)",
            [(path,) for path in distinct_paths],
        )
        path_to_idx = {
            row["path"]: row["idx"] for row in cur.execute("SELECT idx, path FROM catalogues")
        }
        cur.executemany(
            "UPDATE results SET catalogue_idx = ? WHERE filename = ?",
            [(path_to_idx[str(cat)], str(sid)) for sid, cat in source_to_catalogue.items()],
        )
        self._conn.commit()

    def get_catalogue_map(self, source_ids: list[str]) -> dict[str, str]:
        """Return the recorded catalogue path for each given scored source (#506).

        Args:
            source_ids: Source ids to look up.

        Returns:
            Map from ``source_id`` to its catalogue path, containing only the
            ids whose ``catalogue_idx`` has been recorded (others fall back to a
            full catalogue scan in the loader).  An empty input yields an empty
            map.
        """
        wanted = [str(s) for s in source_ids]
        rows = self._select_in_chunks(
            "SELECT r.filename AS filename, c.path AS path "
            "FROM results r JOIN catalogues c ON r.catalogue_idx = c.idx "
            "WHERE r.filename IN ({placeholders})",
            wanted,
        )
        return {row["filename"]: row["path"] for row in rows}

    def set_metadata(self, key: str, value: Any) -> None:
        """Store a metadata key-value pair (JSON-serialised).

        Args:
            key: Metadata key (e.g. ``"model_path"``).
            value: Arbitrary JSON-serialisable value.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO run_metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()

    def set_metadata_batch(self, items: dict[str, Any]) -> None:
        """Store multiple metadata key-value pairs in one transaction.

        Args:
            items: Mapping of metadata keys to JSON-serialisable values.
        """
        self._conn.executemany(
            "INSERT OR REPLACE INTO run_metadata(key, value) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in items.items()],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read operations (UI thread)
    # ------------------------------------------------------------------

    def get_results(
        self,
        sort_by: str = "score_desc",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query results with sorting and pagination.

        Args:
            sort_by: One of ``"score_desc"``, ``"score_asc"``,
                ``"score_mean_dist"``, ``"score_median_dist"``,
                ``"updated_desc"``, ``"updated_asc"``, ``"random"``.
            limit: Maximum number of rows to return.
            offset: Number of rows to skip.

        Returns:
            List of dicts with keys ``filename`` and ``score``.

        Raises:
            ValueError: If *sort_by* is not a recognised sort mode.
        """
        order = _SORT_CLAUSES.get(sort_by)
        if order is None:
            raise ValueError(f"Unknown sort_by={sort_by!r}. Valid options: {list(_SORT_CLAUSES)}")

        # Safe: sort_by is validated against _SORT_CLAUSES allowlist
        rows = self._safe_execute(
            f"""
            SELECT r.filename, r.score
            FROM results r
            ORDER BY {order}
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        return [
            {
                "filename": row["filename"],
                "score": _fp32_decode(row["score"]),
            }
            for row in rows
        ]

    def get_top_results(self, n: int) -> list[dict]:
        """Convenience wrapper: return the *n* highest-scoring results.

        Args:
            n: Number of top results to return.

        Returns:
            List of result dicts sorted by descending score.
        """
        return self.get_results(sort_by="score_desc", limit=n, offset=0)

    def get_count(self) -> int:
        """Return total number of stored results.

        Returns:
            Row count of the results table.
        """
        rows = self._safe_execute("SELECT COUNT(*) AS cnt FROM results")
        return rows[0]["cnt"] if rows else 0

    def get_result_by_filename(self, filename: str) -> dict | None:
        """Look up a single result by its filename.

        Args:
            filename: The source filename to look up.

        Returns:
            Dict with keys ``filename`` and ``score``, or ``None`` if
            not found.
        """
        rows = self._safe_execute(
            """
            SELECT r.filename, r.score
            FROM results r
            WHERE r.filename = ?
            """,
            (filename,),
        )
        if not rows:
            logger.debug("No result found for filename: %s", filename)
            return None
        row = rows[0]
        return {
            "filename": row["filename"],
            "score": _fp32_decode(row["score"]),
        }

    def get_all_scores(self) -> np.ndarray:
        """Return every stored score as a float32 numpy array.

        Loads the entire ``score`` column into memory — not suitable for
        databases with more than a few million rows.

        Returns:
            1-D array of all scores.
        """
        rows = self._safe_execute("SELECT score FROM results")
        if not rows:
            return np.empty(0, dtype=np.float32)
        # Both dtypes are pinned little-endian so the pair can never drift
        # apart.  Native/native would be equally correct — the value arrives
        # from SQLite as a Python int, so there is no on-the-wire byte order to
        # match; mixing the two views is the only thing that corrupts the
        # reinterpretation.
        raw = np.fromiter((r["score"] for r in rows), dtype="<u4", count=len(rows))
        return raw.view("<f4")

    def get_score_range(self) -> tuple[float, float]:
        """Return ``(min_score, max_score)`` across all results.

        Returns:
            Tuple of minimum and maximum score values.

        Raises:
            ValueError: If the database contains no results.
        """
        rows = self._safe_execute("SELECT MIN(score) AS lo, MAX(score) AS hi FROM results")
        if not rows or rows[0]["lo"] is None:
            raise ValueError("No results in database")
        return (_fp32_decode(rows[0]["lo"]), _fp32_decode(rows[0]["hi"]))

    def get_score_histogram(self, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Compute a score histogram entirely in SQL.

        Args:
            bins: Number of histogram bins (default 50).

        Returns:
            ``(counts, bin_edges)`` — *counts* has length *bins*,
            *bin_edges* has length ``bins + 1``.  Empty bins have
            count 0.
        """
        # Use SQL to bin scores into [0, 1] range.
        # Scores are stored as fp32 bit patterns; decode via the fp32 UDF
        # before multiplying by the bin count.  Scores exactly equal to
        # 1.0 are clamped into the last bin.
        rows = self._safe_execute(
            """
            SELECT
                CASE
                    WHEN CAST(fp32(score) * ? AS INTEGER) >= ? THEN ? - 1
                    ELSE CAST(fp32(score) * ? AS INTEGER)
                END AS bin,
                COUNT(*) AS cnt
            FROM results
            GROUP BY bin
            ORDER BY bin
            """,
            (bins, bins, bins, bins),
        )

        counts = np.zeros(bins, dtype=np.int64)
        for row in rows:
            idx = int(row["bin"])
            if 0 <= idx < bins:
                counts[idx] = row["cnt"]

        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        return counts, bin_edges

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def is_processed(self, filename: str) -> bool:
        """Check whether *filename* already has a stored result.

        Args:
            filename: Source filename to check.

        Returns:
            ``True`` if a result row exists for this filename.
        """
        row = self._conn.execute("SELECT 1 FROM results WHERE filename = ?", (filename,)).fetchone()
        return row is not None

    def _select_in_chunks(self, sql_template: str, values: list[str]) -> list[sqlite3.Row]:
        """Run an ``IN``-clause SELECT over *values* in bound-variable-safe chunks.

        *sql_template* must contain a single ``{placeholders}`` field; it is
        filled with the ``?`` markers for each chunk and executed against slices
        of at most :data:`_MAX_SQL_VARIABLES` values.  Rows from all chunks are
        concatenated.  SELECT-only, so it takes no write lock.

        Args:
            sql_template: SQL with a single ``{placeholders}`` slot for the IN clause.
            values: Bound values for the IN clause.

        Returns:
            Matching rows across every chunk.
        """
        rows: list[sqlite3.Row] = []
        for start in range(0, len(values), _MAX_SQL_VARIABLES):
            chunk = values[start : start + _MAX_SQL_VARIABLES]
            placeholders = ",".join("?" * len(chunk))
            rows.extend(
                self._conn.execute(sql_template.format(placeholders=placeholders), chunk).fetchall()
            )
        return rows

    def get_unprocessed(self, filenames: list[str]) -> list[str]:
        """Filter *filenames* to only those without stored results.

        SELECT-only (no temp tables); input order and any duplicate entries are
        preserved.

        Args:
            filenames: Candidate filenames to check.

        Returns:
            Subset of *filenames* that do not yet appear in the
            results table.
        """
        if not filenames:
            return []

        # SELECT-only so this resume check takes no write lock. The previous
        # temp-table approach (CREATE/INSERT/DELETE) opened an IMMEDIATE write
        # transaction (the connection's isolation_level) and contended with the
        # live scoring writer for the single WAL writer slot — a lock-upgrade
        # collision then returns SQLITE_BUSY immediately, which busy_timeout
        # can't wait out, and killed a prediction chunk.
        rows = self._select_in_chunks(
            "SELECT filename FROM results WHERE filename IN ({placeholders})", filenames
        )
        processed = {row["filename"] for row in rows}
        return [f for f in filenames if f not in processed]

    def get_processed_filenames(self) -> set[str]:
        """Return the set of all processed filenames.

        For very large databases this loads all filenames into memory.
        Prefer :meth:`get_unprocessed` for incremental resume checks.

        Returns:
            Set of filenames with stored results.
        """
        rows = self._conn.execute("SELECT filename FROM results").fetchall()
        return {row["filename"] for row in rows}

    # ------------------------------------------------------------------
    # Compatibility validation
    # ------------------------------------------------------------------

    def validate_compatibility(self, cfg: DotMap) -> tuple[bool, str]:
        """Check whether *cfg* matches the stored run metadata.

        Only keys listed in :data:`_COMPAT_KEYS` are compared.  The
        schema-version entry is always present (written on open) but is
        not part of the compatibility check — a schema mismatch is
        already rejected in :meth:`__init__`.

        ``cfg`` is the prediction config; this method calls
        :func:`build_compat_metadata` internally so all DB-related
        metadata extraction lives next to the DB.  Callers that already
        hold a built dict can use :meth:`set_metadata_batch` directly.

        Args:
            cfg: Prediction config with ``model_path``, ``net``, and
                ``normalisation`` populated.

        Returns:
            ``(is_compatible, message)`` — *message* describes the first
            mismatch found, or is empty on success.  A fresh DB (no
            compat keys stored yet) is always compatible.
        """
        cfg_metadata = build_compat_metadata(cfg)
        stored = self.get_metadata()

        for key, label in _COMPAT_KEYS.items():
            if key not in stored:
                logger.debug("Compatibility key %r not in stored metadata; skipping", key)
                continue
            if stored[key] != cfg_metadata[key]:
                return (
                    False,
                    f"{label} mismatch: stored={stored[key]!r}, current={cfg_metadata[key]!r}",
                )

        return True, ""

    def get_metadata(self) -> dict[str, Any]:
        """Return all stored metadata as a dict.

        Returns:
            Mapping of metadata keys to their JSON-deserialised values.
        """
        rows = self._conn.execute("SELECT key, value FROM run_metadata").fetchall()
        result = {row["key"]: json.loads(row["value"]) for row in rows}
        if not result:
            logger.debug("No metadata found in prediction DB")
        return result

    def get_heartbeat(self) -> int | None:
        """Return the unix-epoch timestamp of the last successful commit.

        Refreshed by :meth:`store_results` in the same transaction as
        the scores.  A stale value (more than a few minutes behind the
        wall clock) means the writer subprocess is paused or crashed.

        Returns:
            Integer unix timestamp, or ``None`` if no batch has committed
            since the DB was created.
        """
        rows = self._safe_execute("SELECT value FROM run_metadata WHERE key = ?", (_HEARTBEAT_KEY,))
        if not rows:
            return None
        return int(json.loads(rows[0]["value"]))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Return the path to the database file.

        Returns:
            Database file path.
        """
        return self._db_path

    def close(self) -> None:
        """Close the database connection.

        Acquires ``_close_lock`` to wait for any in-flight
        ``_safe_execute`` calls to finish, then nulls ``_conn`` so
        subsequent reads return ``[]``.  The actual ``conn.close()``
        runs after the lock is released — at that point no thread can
        hold a reference to the connection.

        On the way out:

        * ``PRAGMA wal_checkpoint(TRUNCATE)`` folds the WAL back into
          the main file so a reopened DB does not carry a ``*.wal``
          leftover from the last run.
        * ``PRAGMA optimize`` lets SQLite regenerate statistics on
          indexes that have changed since they were last ANALYZEd.

        A best-effort wrapper swallows per-pragma errors so a write
        lock held by something else (or a broken connection) can't
        block shutdown.
        """
        with self._close_lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                for stmt in ("PRAGMA wal_checkpoint(TRUNCATE)", "PRAGMA optimize"):
                    try:
                        conn.execute(stmt)
                    except sqlite3.Error as exc:
                        logger.debug("{} failed during close: {}", stmt, exc)
        if conn is not None:
            conn.close()
        logger.debug("AnomalyScoreDB closed: {}", self._db_path)

    def _safe_execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute a read query, returning ``[]`` if the connection is closed.

        Holds ``_close_lock`` for the duration of the query so that
        ``close()`` cannot free the C-level sqlite3 handle while a
        thread is mid-execute (which causes a segfault on Linux).

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            List of rows, or empty list if the connection was closed.
        """
        with self._close_lock:
            conn = self._conn
            if conn is None:
                return []
            try:
                return conn.execute(sql, params).fetchall()
            except sqlite3.ProgrammingError:
                return []

    def __enter__(self) -> AnomalyScoreDB:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"AnomalyScoreDB({str(self._db_path)!r})"
