#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Resolve where the live ``predictions.db`` lives and keep the session durable.

WAL-mode SQLite always keeps its ``-wal``/``-shm`` sidecars next to the main
``.db`` file, so the *whole* database must sit on fast local storage to avoid
the WAL bloat a network filesystem (NFS) suffers under a continuous UI reader
(a slow read pins a reader snapshot, no checkpoint can advance past it, and the
WAL grows without bound — it reached 11 GB in practice).

This module is the single source of truth for the live DB path. When
``output_dir`` is on a network filesystem it transparently relocates the DB to a
local-disk scratch directory; :func:`prediction_db_path` is a *pure* function of
the config, so every process (the prediction subprocess, the UI poller, the
orchestrating Session) resolves the same location regardless of call order.
:func:`prepare_local_db` and :func:`snapshot_db_to_session` keep the portable
session directory seeded and durable around a relocated run.
"""

import os
import shutil
import tempfile

from loguru import logger

_DB_FILENAME = "predictions.db"

# Snapshotting back to the (networked) session dir is best-effort per chunk, but a
# sustained failure means the durable copy is going stale while live scores sit
# only on ephemeral local scratch — a pod loss would drop them. Track consecutive
# failures and escalate from warning to error once they pass this many in a row.
_SNAPSHOT_FAILURE_ERROR_THRESHOLD = 3
_consecutive_snapshot_failures = 0

# Filesystem types where per-commit WAL fsync and slow reads cause the bloat.
_NETWORK_FSTYPES = frozenset(
    {"nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse.sshfs", "fuse.glusterfs", "lustre"}
)


def _filesystem_type(path: str) -> str | None:
    """Filesystem type of the deepest ``/proc/mounts`` mount under *path*.

    Returns:
        The mount's type (e.g. ``"nfs4"``), or ``None`` when undeterminable —
        notably on macOS/Windows, which have no ``/proc/mounts``.
    """
    try:
        with open("/proc/mounts") as handle:
            entries = [line.split() for line in handle]
    except OSError:
        # No /proc/mounts (macOS/Windows): we can't classify the mount, so treat
        # it as local and leave the DB in place. Relocation targets the Linux/NFS
        # deployment; a networked volume on those OSes is not detected and keeps
        # the DB in the session dir.
        return None
    real = os.path.realpath(path)
    best_mount = None
    best_type = None
    for parts in entries:
        if len(parts) < 3:
            continue
        mount, fstype = parts[1], parts[2]
        if real == mount or real.startswith(mount.rstrip("/") + "/"):
            if best_mount is None or len(mount) > len(best_mount):
                best_mount, best_type = mount, fstype
    return best_type


def is_network_filesystem(path: str) -> bool:
    """Report whether *path* is backed by a known network filesystem.

    Args:
        path: Filesystem path to classify.

    Returns:
        ``True`` for NFS/CIFS/Lustre/etc., ``False`` for local storage or when
        the type cannot be determined (e.g. non-Linux platforms).
    """
    return _filesystem_type(path) in _NETWORK_FSTYPES


def _local_scratch_dir(output_dir: str) -> str | None:
    """Local-disk scratch dir for *output_dir* (per-session leaf to avoid collisions).

    Returns:
        A local scratch dir when *output_dir* is networked and a local temp
        exists, else ``None`` to keep the DB in *output_dir*.
    """
    if not is_network_filesystem(output_dir):
        return None
    temp_root = tempfile.gettempdir()
    if is_network_filesystem(temp_root):
        # A networked temp dir would just move the problem; keep the DB in place.
        return None
    session_leaf = os.path.basename(os.path.normpath(output_dir)) or "session"
    return os.path.join(temp_root, "anomaly_match_db", session_leaf)


def _effective_db_dir(cfg, output_dir: str | None = None) -> str:
    """Directory the live DB should live in for *cfg*.

    An explicit ``cfg.prediction_db_dir`` wins (and disables auto relocation —
    set it to *output_dir* to pin the DB in the session). Otherwise the DB
    relocates to local scratch when the session dir is networked, else stays put.

    Returns:
        The directory the live ``predictions.db`` should live in.
    """
    explicit = cfg.prediction_db_dir
    # Two distinct guards: it defaults to None (and DotMap.copy() can turn that
    # into an empty DotMap), so `isinstance(str)` rejects the unset case; the
    # truthiness then rejects an explicit "" — only a real, non-empty path
    # overrides auto-relocation.
    if isinstance(explicit, str) and explicit:
        return explicit
    base = output_dir if output_dir else cfg.output_dir
    local = _local_scratch_dir(base)
    return local if local is not None else base


def prediction_db_path(cfg, output_dir: str | None = None) -> str:
    """Return the path to the live ``predictions.db`` for *cfg*.

    Pure function of the config — all writers and readers call this so they
    agree on one location. See the module docstring for the relocation policy.

    Args:
        cfg: Session configuration (DotMap).
        output_dir: Override for the session dir (e.g. a UI chooser selection);
            defaults to ``cfg.output_dir``.

    Returns:
        Absolute-or-relative path to ``predictions.db``.
    """
    return os.path.join(_effective_db_dir(cfg, output_dir), _DB_FILENAME)


def is_relocated(cfg, output_dir: str | None = None) -> bool:
    """Whether the live DB lives outside the session dir (in local scratch).

    Returns:
        ``True`` when relocated to local scratch, else ``False``.
    """
    session_dir = output_dir if output_dir else cfg.output_dir
    return os.path.normpath(_effective_db_dir(cfg, output_dir)) != os.path.normpath(session_dir)


def session_db_path(cfg, output_dir: str | None = None) -> str:
    """Path to the durable ``predictions.db`` snapshot in the session dir.

    This is the portable copy that survives a relocated run (see
    :func:`snapshot_db_to_session`); it equals :func:`prediction_db_path` when
    the DB is not relocated. Centralises the filename so callers don't hard-code
    it.

    Args:
        cfg: Session configuration (DotMap).
        output_dir: Override for the session dir; defaults to ``cfg.output_dir``.

    Returns:
        Path to ``predictions.db`` inside the session directory.
    """
    base = output_dir if output_dir else cfg.output_dir
    return os.path.join(base, _DB_FILENAME)


def _copy_db_files(src_db: str, dst_db: str) -> None:
    """Copy a quiescent SQLite DB (``.db`` plus any non-empty ``-wal``) to *dst_db*.

    Atomicity here is *destination*-side: the ``.db`` is staged to a temp file and
    swapped in with :func:`os.replace`, so a concurrent reader of *dst_db* never
    sees a half-written file. This is not an online backup — a consistent *source*
    snapshot relies on the caller's precondition that no writer is open (the
    prediction subprocess has exited between chunks), which is what keeps the
    ``.db`` and its ``-wal`` mutually consistent while they are copied. A leftover
    non-empty ``-wal`` is carried along so a crash-time, un-checkpointed WAL is
    preserved for recovery, and stale destination sidecars are cleared first to
    avoid a mismatched ``.db``/``-wal`` pair.
    """
    os.makedirs(os.path.dirname(dst_db), exist_ok=True)
    for suffix in ("-wal", "-shm"):
        stale = dst_db + suffix
        if os.path.isfile(stale):
            os.remove(stale)
    tmp = dst_db + ".copytmp"
    shutil.copy2(src_db, tmp)
    os.replace(tmp, dst_db)
    wal = src_db + "-wal"
    carried_wal = os.path.isfile(wal) and os.path.getsize(wal) > 0
    if carried_wal:
        shutil.copy2(wal, dst_db + "-wal")
    logger.debug(
        "Copied prediction DB {} -> {} ({} bytes{}).",
        src_db,
        dst_db,
        os.path.getsize(dst_db),
        ", with -wal" if carried_wal else "",
    )


def prepare_local_db(cfg) -> None:
    """Create the local scratch dir and seed it from any session snapshot.

    Call once before launching prediction subprocesses. No-op when the DB is not
    relocated. The seed lets a fresh pod resume a run whose durable copy lives in
    the (networked) session directory.

    Args:
        cfg: Session configuration; ``cfg.output_dir`` is the session dir.
    """
    if not is_relocated(cfg):
        logger.debug("Prediction DB not relocated; no local scratch to prepare.")
        return
    local = prediction_db_path(cfg)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    logger.info(
        "Live predictions.db relocated to local scratch {} (session dir {} is on a "
        "network filesystem); snapshotting back to the session after each chunk.",
        os.path.dirname(local),
        cfg.output_dir,
    )
    if os.path.isfile(local):
        # Scratch is pod-local and ephemeral, so a present copy is this pod's own
        # in-progress DB — authoritative over the session snapshot. Don't reseed.
        logger.debug("Local prediction DB already present at {}; not reseeding.", local)
        return
    session = session_db_path(cfg)
    if os.path.isfile(session):
        _copy_db_files(session, local)
        logger.info("Seeded local prediction DB from session snapshot {}.", session)


def snapshot_db_to_session(cfg) -> None:
    """Copy the live local DB back into the session dir for durability.

    No-op when the DB is not relocated (it already lives in the session). Safe to
    call between chunks: the chunk subprocess has exited, so the local DB has no
    open writer.

    Args:
        cfg: Session configuration; the DB is copied into ``cfg.output_dir``.
    """
    if not is_relocated(cfg):
        logger.debug("Prediction DB not relocated; session copy is already live.")
        return
    local = prediction_db_path(cfg)
    if not os.path.isfile(local):
        # No local DB yet (no chunk has written one) — nothing to snapshot.
        logger.debug("No local prediction DB at {} to snapshot.", local)
        return
    session = session_db_path(cfg)
    # Best-effort durability: a failed snapshot (e.g. a read-only session dir or
    # a transient NFS error) must not abort the run — the authoritative data is
    # still in the local DB and the next chunk will retry the snapshot. A *run*
    # of failures is escalated to an error so a persistently undurable session is
    # not lost in debug noise.
    global _consecutive_snapshot_failures
    try:
        _copy_db_files(local, session)
        _consecutive_snapshot_failures = 0
        logger.debug("Snapshotted prediction DB to session {}.", session)
    except OSError as exc:
        _consecutive_snapshot_failures += 1
        if _consecutive_snapshot_failures >= _SNAPSHOT_FAILURE_ERROR_THRESHOLD:
            logger.error(
                "Prediction DB snapshot to session {} has failed {} times in a row "
                "(latest: {}). The durable session copy is stale; recent scores live "
                "only on local scratch and would be lost if this pod is reclaimed.",
                session,
                _consecutive_snapshot_failures,
                exc,
            )
        else:
            logger.warning("Could not snapshot prediction DB to session {}: {}", session, exc)
