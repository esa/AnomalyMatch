#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Loguru-sink helper for turning log stream into structured UI progress events.

The training and prediction subprocesses already emit per-batch progress via
``tqdm``; :class:`Session._launch_prediction_subprocess` relays every subprocess
stdout line through ``logger.info("[subprocess] {}", line)`` so the lines flow
into the main kernel's loguru stream (and into the screen's Output widget).

This module lets a UI screen tap that stream read-only — without touching the
background thread that produced the line — to extract ``71/93`` style progress
counts and any ``cutana`` module messages that are useful as spinner captions.
The tap is a loguru sink, so installing and removing it has no effect on the
background work; it simply forks a copy of the log messages to the supplied
callbacks.
"""

from __future__ import annotations

import re
from typing import Callable

from loguru import logger

# Matches ``{desc}: {pct}%|{bar}| {current}/{total} [{elapsed}<{remaining}, {rate}]``.
# Applied with ``re.search`` so it still matches when the line is wrapped in a
# ``[subprocess]`` relay prefix and a loguru-formatted timestamp/level header
# (e.g. ``13:37:42|INFO|[subprocess] 12:34:56 | INFO | x | Streaming cutouts: ...``).
# The ``bar`` portion is matched greedily up to ``|`` so Unicode block
# characters don't confuse the regex; ``desc`` is anchored to the closest
# ``{word}: {percent}%`` so loguru header fragments don't eat it.
_TQDM_RE = re.compile(
    r"(?P<desc>[^|\[\]\n]+?):\s*"
    r"(?P<pct>\d+)%\|[^|]*\|\s*"
    r"(?P<current>\d+)/(?P<total>\d+)"
    r"\s*\[(?P<timing>[^\]]*)\]"
)

# INFO-level hints that make useful spinner sub-captions on the setup
# screen.  We match on content rather than loguru's ``record["name"]``
# because subprocess-relayed lines arrive via Session and so carry the
# Session module name, not ``cutana.*``.  Two families:
#   - Cutana lifecycle messages from ``create_cutouts_direct`` /
#     ``StreamingOrchestrator``.
#   - Our own ``source_validation`` per-catalogue progress lines, added
#     specifically so the setup screen spinner doesn't sit for minutes
#     on billion-row Cutana sources.
_CUTANA_HINT_RE = re.compile(
    r"(?:Creating \d+ cutouts directly|"
    r"Loaded \d+ FITS files|"
    r"Grouped into \d+ unique FITS file sets|"
    r"Generated \d+ cutouts directly|"
    r"streaming \d+ (?:labeled|unlabeled) cutouts|"
    r"Initializing streaming|"
    r"Streaming initialized|"
    r"Validating labels\b|"
    r"Cutana validation complete|"
    r"Building labeled cache\b|"
    # Cumulative cutout progress while (re)building the labeled cache, so the
    # spinner shows an overall count rather than per-catalogue FITS-set noise.
    r"Extracting labeled data\b|"
    # Emitted by LabeledDataCache.get_raw_images_by_id during the preview's
    # (NFS-bound) cache read; keep in sync with that emitter's _CACHE_READ_DESC.
    r"Loading labeled cache\b|"
    r"Loading preview: reading catalogue\b|"
    # Prediction (scoring) startup phases — the otherwise-silent stretch
    # between "Starting scoring..." and the first "Processing batches" tick:
    # orchestrator init, model load, GPU warmup, and the first FITS-tile
    # stream.  Surfaced so scoring shows signs of life while it spins up.
    r"Creating Cutana orchestrator|"
    r"Cutana orchestrator streaming mode|"
    r"Available batches in cutana|"
    r"Loading model with|"
    r"Warming up GPU|"
    r"Streaming first cutout batch)",
    re.IGNORECASE,
)

# Strips a relayed subprocess loguru header — ``[subprocess] <ts> | LEVEL |
# name:func:line - `` — so a hint shown in a status label reads as just the
# message.  Lines logged in the UI kernel (no relay) have no such header and
# pass through untouched.
#
# This depends on the subprocess's *default* loguru format (see
# ``setup_prediction_logging`` in subprocess_scripts/prediction_utils.py): a
# timestamp, a level, and a ``name:function:line`` location separated by
# `` | ``, then `` - `` before the message.  If that format ever changes the
# match fails and ``clean_hint_text`` returns the still-prefixed line — a
# known breakage point to update here in lock-step.
_LOG_HEADER_RE = re.compile(
    r"^\[subprocess\]\s*\d{4}-\d\d-\d\d[ T][\d:.]+\s*\|\s*\w+\s*\|\s*[^|]+?\s-\s"
)


def clean_hint_text(text: str) -> str:
    """Return *text* with any relayed subprocess loguru header removed.

    Args:
        text: A log line matched by :func:`looks_like_cutana_status`, which
            for subprocess-relayed lines still carries the child's
            ``time | LEVEL | location - `` prefix.

    Returns:
        Just the human-readable message, suitable for a status label.
    """
    return _LOG_HEADER_RE.sub("", text).strip()


def parse_tqdm_line(text: str) -> dict | None:
    """Extract a tqdm progress event from a single log line.

    Args:
        text: Raw log message text (without loguru formatting).

    Returns:
        Dict with ``desc``, ``percent``, ``current``, ``total``, ``timing``
        keys when *text* looks like a tqdm status line; ``None`` otherwise.
    """
    m = _TQDM_RE.search(text)
    if m is None:
        return None
    return {
        "desc": m.group("desc").strip(),
        "percent": int(m.group("pct")),
        "current": int(m.group("current")),
        "total": int(m.group("total")),
        "timing": m.group("timing").strip(),
    }


def looks_like_cutana_status(text: str) -> bool:
    """Return True when *text* matches one of the known Cutana status hints."""
    return _CUTANA_HINT_RE.search(text) is not None


class LogProgressTap:
    """A loguru sink that forks log messages to progress-aware callbacks.

    Install once when a long-running operation starts, uninstall on
    completion.  The sink runs in whichever thread loguru dispatches
    from (usually the thread that produced the message), so callbacks
    must be cheap and widget updates must be safe to perform from a
    non-UI thread — which is the case for ipywidgets attribute writes.

    Args:
        on_tqdm: Optional callback invoked with the parsed dict from
            :func:`parse_tqdm_line` on every matching line.
        on_cutana_hint: Optional callback invoked with the raw message
            text when the line matches :func:`looks_like_cutana_status`.
    """

    def __init__(
        self,
        *,
        on_tqdm: Callable[[dict], None] | None = None,
        on_cutana_hint: Callable[[str], None] | None = None,
    ) -> None:
        self._on_tqdm = on_tqdm
        self._on_cutana_hint = on_cutana_hint
        self._sink_id: int | None = None

    def install(self) -> None:
        """Attach the sink to the shared loguru logger.

        Calling this twice without an intervening :meth:`uninstall` is a
        no-op so callers don't need to track state.
        """
        if self._sink_id is not None:
            return
        self._sink_id = logger.add(self._sink, level="INFO", format="{message}")

    def uninstall(self) -> None:
        """Detach the sink.  Safe to call multiple times."""
        if self._sink_id is None:
            return
        try:
            logger.remove(self._sink_id)
        except ValueError:
            # Sink already removed (e.g. via logger.remove() elsewhere).
            pass
        self._sink_id = None

    def _sink(self, message) -> None:  # loguru Message
        text = message.record["message"]

        if self._on_tqdm is not None:
            event = parse_tqdm_line(text)
            if event is not None:
                try:
                    self._on_tqdm(event)
                except Exception as exc:  # noqa: BLE001 — must never block sink
                    logger.debug("progress tap on_tqdm callback failed: {}", exc)
                return

        if self._on_cutana_hint is not None and looks_like_cutana_status(text):
            try:
                self._on_cutana_hint(text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("progress tap on_cutana_hint callback failed: {}", exc)
