#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for training_process.py helper functions and config constants."""

import json
import sys

from anomaly_match.utils.get_default_cfg import get_default_cfg

# ── Config constants ─────────────────────────────────────────────


class TestTrainingConfigConstants:
    """Verify training pool cap constants exist and have sane defaults."""

    def test_unlabeled_pool_cap_exists(self):
        cfg = get_default_cfg()
        assert cfg.unlabeled_pool_cap == 20_000

    def test_unlabeled_pool_cap_hires_exists(self):
        cfg = get_default_cfg()
        assert cfg.unlabeled_pool_cap_hires == 10_000

    def test_hires_threshold_exists(self):
        cfg = get_default_cfg()
        assert cfg.unlabeled_pool_hires_threshold == 200

    def test_hires_cap_is_smaller_than_lores(self):
        cfg = get_default_cfg()
        assert cfg.unlabeled_pool_cap_hires < cfg.unlabeled_pool_cap


# ── _write_progress ──────────────────────────────────────────────


class TestWriteProgress:
    """Tests for the _write_progress helper."""

    def test_writes_jsonl(self, tmp_path):
        # Import from scripts/ needs sys.path adjustment
        sys.path.insert(0, str(tmp_path.parent.parent.parent / "scripts"))
        try:
            from training_process import _write_progress
        finally:
            sys.path.pop(0)

        progress_file = str(tmp_path / "progress.jsonl")
        _write_progress(progress_file, {"status": "training", "iteration": 1, "total": 10})
        _write_progress(progress_file, {"status": "done", "elapsed": 5.0})

        with open(progress_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["status"] == "training"
        assert json.loads(lines[1])["status"] == "done"

    def test_appends_not_overwrites(self, tmp_path):
        sys.path.insert(0, str(tmp_path.parent.parent.parent / "scripts"))
        try:
            from training_process import _write_progress
        finally:
            sys.path.pop(0)

        progress_file = str(tmp_path / "progress.jsonl")
        for i in range(5):
            _write_progress(progress_file, {"iteration": i})

        with open(progress_file) as f:
            lines = f.readlines()
        assert len(lines) == 5


# ── N_unlabeled computation ──────────────────────────────────────


class TestUnlabeledPoolComputation:
    """Tests for the N_unlabeled formula used by training_process.py."""

    def test_lores_uses_larger_cap(self):
        cfg = get_default_cfg()
        cfg.normalisation.image_size = [64, 64]
        cap = (
            cfg.unlabeled_pool_cap_hires
            if max(cfg.normalisation.image_size) > cfg.unlabeled_pool_hires_threshold
            else cfg.unlabeled_pool_cap
        )
        assert cap == 20_000

    def test_hires_uses_smaller_cap(self):
        cfg = get_default_cfg()
        cfg.normalisation.image_size = [256, 256]
        cap = (
            cfg.unlabeled_pool_cap_hires
            if max(cfg.normalisation.image_size) > cfg.unlabeled_pool_hires_threshold
            else cfg.unlabeled_pool_cap
        )
        assert cap == 10_000

    def test_formula_respects_cap(self):
        cfg = get_default_cfg()
        cfg.normalisation.image_size = [64, 64]
        cfg.num_train_iter = 200
        # batch_size=16, uratio=5, num_train_iter=200 → 16*5*200 = 16000
        n_unlabeled = min(cfg.batch_size * cfg.uratio * cfg.num_train_iter, cfg.unlabeled_pool_cap)
        assert n_unlabeled == 16_000

    def test_formula_capped_when_exceeding(self):
        cfg = get_default_cfg()
        cfg.normalisation.image_size = [64, 64]
        cfg.num_train_iter = 600
        # batch_size=16, uratio=5, num_train_iter=600 → 48000 > cap=20000
        n_unlabeled = min(cfg.batch_size * cfg.uratio * cfg.num_train_iter, cfg.unlabeled_pool_cap)
        assert n_unlabeled == 20_000
