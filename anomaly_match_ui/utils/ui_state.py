#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Per-user UI state persistence for AnomalyMatch.

Stores small bits of UX state (chooser last-browsed directories,
window-level preferences) in a JSON file under the user's home
directory.  Anything that the user expects to "remember" between
notebook sessions but that is not part of the current run's config
belongs here.

Schema (``schema_version=1``):

```json
{
    "schema_version": 1,
    "last_browsed_dirs": {
        "model_chooser": "/path/last/visited",
        "source_chooser": "/path/last/visited",
        ...
    },
    "normalisation_settings": {
        "training_setup": {
            "extensions": ["VIS", "NIR-H"],
            "settings": {"normalisation_method": 3, "image_size": [150, 150], ...}
        }
    }
}
```

Storage location: ``~/.config/anomalymatch/ui_state.json``.  We use
the same path on every platform so the test suite and behaviour stay
cross-platform consistent without pulling in ``platformdirs`` for one
small dotfile.

Override location: set ``ANOMALYMATCH_CONFIG_DIR`` to point the state
file at a different directory (the same hook the test suite uses to
keep its writes in ``tmp_path``).  Useful for portable installs and
for users who prefer a non-default state location.

Failure mode: the helpers swallow any I/O or JSON error and return
``None`` / no-op.  Persistence is a UX nicety — a corrupt state file
must never crash the screen build path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

_SCHEMA_VERSION = 1
_STATE_FILENAME = "ui_state.json"
_STATE_DIRNAME = "anomalymatch"
_NORMALISATION_KEY = "normalisation_settings"
CONFIG_DIR_ENV_VAR = "ANOMALYMATCH_CONFIG_DIR"


def _state_path() -> Path:
    """Return the path to the UI state JSON file.

    Honours :data:`CONFIG_DIR_ENV_VAR` when set so tests and portable
    installs can redirect persistence away from the user's home.
    """
    config_dir_override = os.environ.get(CONFIG_DIR_ENV_VAR)
    if config_dir_override:
        return Path(config_dir_override) / _STATE_FILENAME
    return Path.home() / ".config" / _STATE_DIRNAME / _STATE_FILENAME


def _load() -> dict:
    """Load the UI state dict, returning an empty dict on any failure.

    Returns:
        Parsed JSON object, or ``{}`` if the file is missing, unreadable,
        contains invalid JSON, or doesn't decode to a dict.
    """
    path = _state_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read UI state at {}: {}", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.debug("UI state at {} is not a dict; ignoring", path)
        return {}
    return data


def _save(state: dict) -> None:
    """Write *state* to the UI state JSON file.

    Serialises to a string first and writes through a temporary file in the
    same directory, replaced into place atomically.  Writing straight into the
    real file would leave truncated JSON behind if serialisation failed
    part-way (a value no encoder handles) or the process died mid-write — and
    that half-written file is what every later read has to cope with.

    Best-effort; logs at debug level on any failure so a read-only home
    directory or full disk cannot block the chooser callback.  ``TypeError`` and
    ``ValueError`` are caught alongside ``OSError`` because the state dict comes
    from callers, and a non-serialisable value in it is a bug in the caller, not
    a reason to take down the screen that triggered the save.
    """
    path = _state_path()
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        # Serialise before touching the filesystem: an encoding failure then
        # leaves no file at all rather than a partial one.
        payload = json.dumps(state, indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        logger.debug("Could not write UI state to {}: {}", path, exc)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove partial UI state file {}", tmp_path)


def get_last_browsed_dir(purpose: str) -> str | None:
    """Return the last directory the user finalised for *purpose*.

    Args:
        purpose: A stable key identifying the chooser site (e.g.
            ``"model_chooser"``, ``"source_chooser"``).

    Returns:
        The persisted directory path, or ``None`` if nothing has been
        recorded for *purpose* yet, or the recorded path no longer
        exists on disk.
    """
    state = _load()
    dirs = state.get("last_browsed_dirs", {})
    candidate = dirs.get(purpose)
    if not candidate:
        return None
    if not os.path.isdir(candidate):
        return None
    return candidate


def set_last_browsed_dir(purpose: str, path: str) -> None:
    """Record *path* as the last directory the user finalised for *purpose*.

    *path* may point at a file or a directory; in both cases the
    parent directory is what gets stored, since that is what the
    chooser should restore on next open.

    Args:
        purpose: A stable key identifying the chooser site.
        path: The selection the user just confirmed (file or directory).
    """
    if not path:
        return
    if os.path.isdir(path):
        directory = path
    else:
        directory = os.path.dirname(path)
    # ipyfilechooser appends an OS separator when the user clicks
    # Select on a folder (its ``selected`` is ``path + sep + filename``
    # with an empty filename), so normalise away trailing separators
    # before storing — otherwise round-trip comparisons fail.
    directory = os.path.normpath(directory) if directory else directory
    if not directory:
        return

    state = _load()
    state["schema_version"] = _SCHEMA_VERSION
    dirs = state.setdefault("last_browsed_dirs", {})
    dirs[purpose] = directory
    _save(state)


def get_normalisation_settings(purpose: str) -> dict | None:
    """Return the normalisation settings last confirmed for *purpose*.

    Args:
        purpose: A stable key identifying the screen (e.g. ``"training_setup"``).

    Returns:
        ``{"settings": {...}, "extensions": [...]}`` exactly as recorded by
        :func:`set_normalisation_settings`, or ``None`` when nothing has been
        recorded yet or the stored record does not have that shape.
        ``extensions`` names the input channels the settings were chosen
        against; a caller must not apply a stored ``channel_combination``
        matrix to a source exposing a different set of channels.
    """
    state = _load()
    records = state.get(_NORMALISATION_KEY, {})
    record = records.get(purpose)
    if record is None:
        return None
    if not isinstance(record, dict):
        logger.debug("Normalisation state for {} is not a dict; ignoring", purpose)
        return None
    settings = record.get("settings")
    extensions = record.get("extensions")
    if not isinstance(settings, dict) or not isinstance(extensions, list):
        logger.debug("Normalisation state for {} is malformed; ignoring", purpose)
        return None
    return {"settings": settings, "extensions": extensions}


def set_normalisation_settings(purpose: str, settings: dict, extensions: list[str]) -> None:
    """Record the normalisation settings the user confirmed for *purpose*.

    Args:
        purpose: A stable key identifying the screen.
        settings: Normalisation config dict.  Values must already be
            JSON-serialisable — run a widget config through
            ``serialisable_config`` first so numpy matrices arrive as
            nested lists.
        extensions: Input-channel names the channel-combination matrix was
            built against, stored so a later restore can tell whether that
            matrix still describes the selected source.
    """
    state = _load()
    state["schema_version"] = _SCHEMA_VERSION
    records = state.setdefault(_NORMALISATION_KEY, {})
    records[purpose] = {"settings": dict(settings), "extensions": list(extensions)}
    _save(state)
