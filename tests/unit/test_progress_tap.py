#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for ``anomaly_match_ui.utils.progress_tap``."""

from loguru import logger

from anomaly_match_ui.utils.progress_tap import (
    LogProgressTap,
    clean_hint_text,
    looks_like_cutana_status,
    parse_tqdm_line,
)


class TestParseTqdmLine:
    def test_subprocess_prefix_stripped(self):
        event = parse_tqdm_line(
            "[subprocess] Processing batches:  76%|#######| 71/93 [05:49<01:45,  4.79s/it]"
        )
        assert event == {
            "desc": "Processing batches",
            "percent": 76,
            "current": 71,
            "total": 93,
            "timing": "05:49<01:45,  4.79s/it",
        }

    def test_no_prefix(self):
        event = parse_tqdm_line("Streaming cutouts:   0%|   | 0/4 [01:02<?, ?batch/s]")
        assert event is not None
        assert event["desc"] == "Streaming cutouts"
        assert event["current"] == 0
        assert event["total"] == 4

    def test_unicode_bar_characters(self):
        event = parse_tqdm_line("foo:  50%|████████▌         | 2/4 [00:01<00:01,  2.0it/s]")
        assert event is not None
        assert event["desc"] == "foo"
        assert event["current"] == 2
        assert event["total"] == 4

    def test_loguru_prefixed_subprocess_line(self):
        """When training_screen relays subprocess stderr via loguru, each line
        arrives wrapped in a loguru header + ``[subprocess]`` + the
        subprocess's own loguru header.  The tap must still extract the
        tqdm counts buried inside."""
        line = (
            "13:37:42|AnomalyMatch-INFO| [subprocess] 12:34:56 | INFO | "
            "am.datasets | Streaming cutouts:  50%|#####| 2/4 [00:05<00:05, 2.00s/it]"
        )
        event = parse_tqdm_line(line)
        assert event is not None
        assert event["desc"] == "Streaming cutouts"
        assert event["current"] == 2
        assert event["total"] == 4

    def test_non_tqdm_line_returns_none(self):
        assert parse_tqdm_line("just some text") is None
        assert parse_tqdm_line("") is None
        assert parse_tqdm_line("ERROR: something went wrong") is None

    def test_cutana_source_streaming_line_is_not_tqdm(self):
        """Cutana's own info logs don't use tqdm formatting."""
        line = "CutanaSource(lazy): streaming 4000 unlabeled cutouts (from 8 FITS tiles)"
        assert parse_tqdm_line(line) is None


class TestLooksLikeCutanaStatus:
    def test_matches_known_hints(self):
        assert looks_like_cutana_status("Creating 9 cutouts directly (in-process)")
        assert looks_like_cutana_status("Loaded 3 FITS files (mmap)")
        assert looks_like_cutana_status("Grouped into 2 unique FITS file sets")
        assert looks_like_cutana_status("Generated 9 cutouts directly")
        assert looks_like_cutana_status(
            "CutanaSource(lazy): streaming 4000 unlabeled cutouts (from 8 FITS tiles)"
        )
        assert looks_like_cutana_status("Initializing streaming: batch_size=100")
        assert looks_like_cutana_status("Streaming initialized: 10 internal batches")
        # Our own source_validation.py progress lines must also flow
        # through to the setup-screen spinner; the whole point of the tap
        # was that these phases had no signal before.
        assert looks_like_cutana_status("Validating labels against 12 Cutana catalogue(s)")
        assert looks_like_cutana_status(
            "Validating labels: catalogue 3/12 (eu_q1_cat.parquet) — 1247/4000 matched so far"
        )
        assert looks_like_cutana_status(
            "Cutana validation complete: 4000/4000 IDs matched across 12 catalogue(s)"
        )
        # Cumulative cutout progress while rebuilding the labeled cache.
        assert looks_like_cutana_status(
            "Extracting labeled data: 320/1097 cutouts (catalogue 48/52: q1.parquet, 6 source(s))"
        )
        # Labeled-cache read heartbeat (LabeledDataCache.get_raw_images_by_id),
        # surfaced so the preview spinner isn't silent during the NFS read.
        assert looks_like_cutana_status("Loading labeled cache: 320/1097 cutouts")
        # Prediction (scoring) startup phases — the silent "Starting scoring..."
        # stretch before the first "Processing batches" tick.
        assert looks_like_cutana_status("Creating Cutana orchestrator, streaming from /x")
        assert looks_like_cutana_status("Cutana orchestrator streaming mode initialized")
        assert looks_like_cutana_status("Available batches in cutana: 33")
        assert looks_like_cutana_status("Loading model with following configuration:")
        assert looks_like_cutana_status("Warming up GPU for inference (cuDNN autotuning)...")
        assert looks_like_cutana_status(
            "Streaming first cutout batch (reading FITS tiles over the network)..."
        )


class TestCleanHintText:
    def test_strips_relayed_subprocess_loguru_header(self):
        line = (
            "[subprocess] 2026-06-22 16:20:36.920 | INFO     | "
            "prediction_utils:load_model:366 - Loading model with following configuration:"
        )
        assert clean_hint_text(line) == "Loading model with following configuration:"

    def test_strips_header_for_main_module_phase(self):
        line = (
            "[subprocess] 2026-06-22 16:20:47.000 | INFO     | __main__:main:138 - "
            "Warming up GPU for inference (cuDNN autotuning)..."
        )
        assert clean_hint_text(line) == "Warming up GPU for inference (cuDNN autotuning)..."

    def test_clean_kernel_line_passes_through(self):
        # UI-kernel lines (no relay/header) must be left untouched.
        assert (
            clean_hint_text("Loading labeled cache: 320/1097 cutouts")
            == "Loading labeled cache: 320/1097 cutouts"
        )

    def test_ignores_unrelated_lines(self):
        assert not looks_like_cutana_status("Training iteration 150/200")
        assert not looks_like_cutana_status("")
        assert not looks_like_cutana_status("Some arbitrary log line")


class TestLogProgressTap:
    def test_fires_tqdm_callback(self):
        hits = []
        tap = LogProgressTap(on_tqdm=lambda e: hits.append(e))
        tap.install()
        try:
            logger.info("Processing batches:  50%|#####| 5/10 [00:05<00:05, 1.00it/s]")
        finally:
            tap.uninstall()

        assert len(hits) == 1
        assert hits[0]["current"] == 5
        assert hits[0]["total"] == 10

    def test_fires_cutana_hint_callback(self):
        hits = []
        tap = LogProgressTap(on_cutana_hint=lambda t: hits.append(t))
        tap.install()
        try:
            logger.info("Generated 9 cutouts directly")
        finally:
            tap.uninstall()

        assert len(hits) == 1
        assert "Generated 9" in hits[0]

    def test_tqdm_takes_precedence_over_cutana_hint(self):
        """A tqdm match must not also fire the cutana-hint callback."""
        tqdm_hits, hint_hits = [], []
        tap = LogProgressTap(
            on_tqdm=lambda e: tqdm_hits.append(e),
            on_cutana_hint=lambda t: hint_hits.append(t),
        )
        tap.install()
        try:
            # The tqdm description happens to contain a cutana phrase,
            # but tqdm matching is checked first.
            logger.info("Streaming cutouts:  50%|#####| 2/4 [00:01<00:01,  2.00s/it]")
        finally:
            tap.uninstall()

        assert len(tqdm_hits) == 1
        assert hint_hits == []

    def test_non_matching_lines_dropped(self):
        hits = []
        tap = LogProgressTap(
            on_tqdm=lambda e: hits.append(("t", e)),
            on_cutana_hint=lambda t: hits.append(("h", t)),
        )
        tap.install()
        try:
            logger.info("unrelated message")
            logger.info("another unrelated thing")
        finally:
            tap.uninstall()

        assert hits == []

    def test_install_is_idempotent(self):
        tap = LogProgressTap(on_tqdm=lambda e: None)
        tap.install()
        first = tap._sink_id
        tap.install()  # second install must not add a second sink
        assert tap._sink_id == first
        tap.uninstall()

    def test_uninstall_is_idempotent(self):
        tap = LogProgressTap(on_tqdm=lambda e: None)
        tap.install()
        tap.uninstall()
        tap.uninstall()  # second uninstall must not error
        assert tap._sink_id is None

    def test_callback_exception_does_not_crash_sink(self):
        """A callback raising must not propagate into the logger."""
        tap = LogProgressTap(on_tqdm=lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
        tap.install()
        try:
            # Must not raise — the sink swallows callback errors.
            logger.info("Processing batches:  50%|#####| 5/10 [00:05<00:05, 1.00it/s]")
        finally:
            tap.uninstall()
