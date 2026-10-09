#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Resolve the starting directory and pre-selected file for setup-screen choosers.

Shared by :class:`TrainingSetupScreen` and :class:`PredictionSetupScreen`, which
otherwise carried two copies of the same precedence rules.
"""

import os

from anomaly_match_ui.utils import ui_state


def resolved_dir(cfg_value: str | None, *, purpose: str) -> str | None:
    """Resolve a directory from persisted state or config, without a fallback.

    Resolution order (#429):

    1. ``ui_state.get_last_browsed_dir(purpose)`` — the directory the user last
       clicked Select in.  Wins over the cfg value because cfg defaults (e.g.
       ``tests/test_data/grayscale``) would otherwise clobber the user's actual
       recent intent on every fresh kernel start.
    2. ``cfg_value`` — used when no persisted dir exists yet, e.g. on first run
       of a notebook with a manually-set ``data_dir``.

    Kept separate from :func:`initial_dir` so callers can tell a real
    resolution from the cwd fallback: pre-selecting the cwd would silently
    adopt it as the user's choice.

    Args:
        cfg_value: A path from config (file or directory), or ``None``.
        purpose: Stable key identifying the chooser site so we can look up its
            persisted last-browsed directory.

    Returns:
        An existing directory, or ``None`` when neither source resolves to one.
    """
    persisted = ui_state.get_last_browsed_dir(purpose)
    if persisted:
        return persisted
    if cfg_value and os.path.isdir(cfg_value):
        return cfg_value
    if cfg_value and os.path.isfile(cfg_value):
        return os.path.dirname(cfg_value)
    return None


def initial_dir(cfg_value: str | None, *, purpose: str) -> str:
    """Derive a starting directory for a chooser, falling back to the cwd.

    Args:
        cfg_value: A path from config (file or directory), or ``None``.
        purpose: Stable key identifying the chooser site so we can look up its
            persisted last-browsed directory.

    Returns:
        :func:`resolved_dir`'s result, or ``os.getcwd()`` when it is ``None``.
    """
    return resolved_dir(cfg_value, purpose=purpose) or os.getcwd()


def initial_file(
    cfg_value: str | None, *, purpose: str, is_shipped_default: bool = False
) -> tuple[str, str]:
    """Resolve the starting directory and pre-selected filename for a chooser.

    A configured file that exists wins outright: the chooser opens in *its*
    directory with it pre-selected, because a chooser that picks a file must
    display the file the screen will then use — ``_resolve_label_path`` loads
    ``cfg.label_file`` and ``_on_start`` writes ``cfg.model_path`` /
    ``cfg.metadata_file`` regardless of what the widget shows.  That reverses
    :func:`initial_dir`'s precedence (#429), which stands for the directory
    choosers.

    The shipped default is not something the user configured, so it keeps #429's
    order: the remembered folder wins, and the default's filename is pre-selected
    only if a file of that name really exists there.  Without this the bundled
    test-data ``labeled_data.csv``, which always exists, would replace the user's
    last-picked labels on every fresh kernel.

    Nothing is pre-selected when the configured file is missing or unset: the
    chooser opens where the user was last browsing and the panel reports the
    field as unfilled, which is again what the caller acts on.

    Args:
        cfg_value: A file path from config, or ``None``.
        purpose: Stable key identifying the chooser site, used to look up its
            persisted last-browsed directory when cfg resolves to nothing.
        is_shipped_default: Whether *cfg_value* is the bundled default rather than
            a path the user configured (see
            ``BackendInterface.is_shipped_default_path``).

    Returns:
        Tuple of ``(directory, filename)``.  ``filename`` is empty when no real
        file resolves; callers pass ``select_default=bool(filename)``.
    """
    if is_shipped_default and cfg_value:
        directory = initial_dir(cfg_value, purpose=purpose)
        name = os.path.basename(cfg_value)
        return directory, name if os.path.isfile(os.path.join(directory, name)) else ""
    if cfg_value and os.path.isfile(cfg_value):
        resolved = os.path.abspath(cfg_value)
        return os.path.dirname(resolved), os.path.basename(resolved)
    return initial_dir(cfg_value, purpose=purpose), ""
