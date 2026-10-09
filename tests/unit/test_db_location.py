#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the prediction DB location/relocation helpers."""

import os

import pytest
from dotmap import DotMap

from anomaly_match.prediction import (
    AnomalyScoreDB,
    db_location,
    is_relocated,
    prediction_db_path,
    prepare_local_db,
    session_db_path,
    snapshot_db_to_session,
)


@pytest.fixture
def nfs_session(tmp_path, monkeypatch):
    """A session dir reported as NFS, with local scratch under *tmp_path*.

    Returns the (cfg, session_dir, scratch_root) so tests can assert relocation
    without a real network filesystem.
    """
    session_dir = tmp_path / "sessions" / "MyRun_TEST"
    session_dir.mkdir(parents=True)
    scratch_root = tmp_path / "local_tmp"
    scratch_root.mkdir()

    real_fstype = db_location._filesystem_type

    def fake_fstype(path):
        # Treat the session subtree as NFS; everything else keeps its real type.
        if str(os.path.realpath(path)).startswith(str(session_dir)):
            return "nfs4"
        return real_fstype(path)

    monkeypatch.setattr(db_location, "_filesystem_type", fake_fstype)
    monkeypatch.setattr(db_location.tempfile, "gettempdir", lambda: str(scratch_root))

    cfg = DotMap(_dynamic=False)
    cfg.prediction_db_dir = None
    cfg.output_dir = str(session_dir)
    return cfg, str(session_dir), str(scratch_root)


def test_local_session_not_relocated(tmp_path):
    """A local (non-network) session dir keeps the DB in place."""
    cfg = DotMap(_dynamic=False)
    cfg.prediction_db_dir = None
    cfg.output_dir = str(tmp_path)
    assert not is_relocated(cfg)
    assert prediction_db_path(cfg) == os.path.join(str(tmp_path), "predictions.db")


def test_explicit_dir_overrides(tmp_path):
    """An explicit prediction_db_dir wins and counts as relocated."""
    cfg = DotMap(_dynamic=False)
    cfg.prediction_db_dir = str(tmp_path / "custom")
    cfg.output_dir = str(tmp_path / "session")
    assert prediction_db_path(cfg) == os.path.join(str(tmp_path / "custom"), "predictions.db")
    assert is_relocated(cfg)


def test_session_db_path_always_in_session_dir(tmp_path):
    """session_db_path stays in the session dir even when the live DB is relocated."""
    cfg = DotMap(_dynamic=False)
    cfg.prediction_db_dir = str(tmp_path / "custom")  # relocate the live DB
    cfg.output_dir = str(tmp_path / "session")
    assert session_db_path(cfg) == os.path.join(str(tmp_path / "session"), "predictions.db")
    # When not relocated, the durable snapshot and the live DB coincide.
    cfg.prediction_db_dir = None
    cfg.output_dir = str(tmp_path / "local")
    assert session_db_path(cfg) == prediction_db_path(cfg)


def test_nfs_session_relocates_to_local_scratch(nfs_session):
    """A network session dir relocates the DB to local scratch."""
    cfg, session_dir, scratch_root = nfs_session
    assert is_relocated(cfg)
    live = prediction_db_path(cfg)
    assert live.startswith(scratch_root)
    assert not live.startswith(session_dir)
    # Per-session leaf prevents collisions between concurrent sessions.
    assert "MyRun_TEST" in live


def test_network_tempdir_disables_relocation(nfs_session, monkeypatch):
    """If the temp root is itself networked, relocation would not help — stay put."""
    cfg, session_dir, _ = nfs_session
    monkeypatch.setattr(db_location, "_filesystem_type", lambda _p: "nfs4")
    assert not is_relocated(cfg)
    assert prediction_db_path(cfg) == os.path.join(session_dir, "predictions.db")


def test_prepare_seed_write_snapshot_resume(nfs_session):
    """Full lifecycle: prepare, write, snapshot, fresh-pod resume via seed."""
    cfg, session_dir, _ = nfs_session
    live = prediction_db_path(cfg)
    session_db = os.path.join(session_dir, "predictions.db")

    prepare_local_db(cfg)  # fresh run: no snapshot to seed from
    with AnomalyScoreDB(live) as db:
        db.store_results([("a", 0.1), ("b", 0.9)])
    snapshot_db_to_session(cfg)
    assert os.path.isfile(session_db)
    with AnomalyScoreDB(session_db) as snap:
        assert snap.get_processed_filenames() == {"a", "b"}

    # Simulate a fresh pod: scratch wiped, durable session snapshot remains.
    import shutil

    shutil.rmtree(os.path.dirname(live))
    assert not os.path.isfile(live)

    prepare_local_db(cfg)  # seeds local DB from the session snapshot
    assert os.path.isfile(live)
    with AnomalyScoreDB(live) as db:
        assert db.get_processed_filenames() == {"a", "b"}
        db.store_results([("c", 0.7)])
    snapshot_db_to_session(cfg)
    with AnomalyScoreDB(session_db) as snap:
        assert snap.get_processed_filenames() == {"a", "b", "c"}


def test_snapshot_failure_run_escalates_to_error(nfs_session, monkeypatch):
    """A run of snapshot failures escalates from warning to error, and a success resets it."""
    cfg, _, _ = nfs_session
    prepare_local_db(cfg)  # create the local scratch dir + DB to snapshot from
    live = prediction_db_path(cfg)
    with AnomalyScoreDB(live) as db:
        db.store_results([("a", 0.1)])

    monkeypatch.setattr(db_location, "_consecutive_snapshot_failures", 0)
    real_copy = db_location._copy_db_files

    levels = []

    class _RecordingLogger:
        def __getattr__(self, level):
            def _log(_message, *_args, **_kwargs):
                levels.append(level)

            return _log

    monkeypatch.setattr(db_location, "logger", _RecordingLogger())

    def _boom(_src, _dst):
        raise OSError("nfs unavailable")

    monkeypatch.setattr(db_location, "_copy_db_files", _boom)

    # Threshold is 3: the first two failures warn, the third escalates to error.
    snapshot_db_to_session(cfg)
    snapshot_db_to_session(cfg)
    assert levels == ["warning", "warning"]
    snapshot_db_to_session(cfg)
    assert levels[-1] == "error"

    # A successful snapshot clears the streak, so a later failure warns again.
    monkeypatch.setattr(db_location, "_copy_db_files", real_copy)
    snapshot_db_to_session(cfg)
    assert db_location._consecutive_snapshot_failures == 0
    monkeypatch.setattr(db_location, "_copy_db_files", _boom)
    snapshot_db_to_session(cfg)
    assert levels[-1] == "warning"


def test_snapshot_and_prepare_are_noops_when_local(tmp_path):
    """On a local session dir, prepare/snapshot do nothing and the path is stable."""
    cfg = DotMap(_dynamic=False)
    cfg.prediction_db_dir = None
    cfg.output_dir = str(tmp_path)
    prepare_local_db(cfg)  # no-op
    with AnomalyScoreDB(prediction_db_path(cfg)) as db:
        db.store_results([("x", 0.5)])
    snapshot_db_to_session(cfg)  # no-op — already in the session dir
    # No stray copy created elsewhere; the only DB is the in-session one.
    assert os.path.isfile(os.path.join(str(tmp_path), "predictions.db"))
