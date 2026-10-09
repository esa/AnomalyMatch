#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
r"""Make a tqdm meter emit one newline-terminated line per refresh.

The training and prediction subprocesses surface progress to the UI by writing
to stderr; the parent relays each stderr line through
``logger.info("[subprocess] {}")`` and the training screen reads it with
``readline()``, which splits only on ``\n``.

tqdm redraws its meter in place with carriage returns (``\r``) and writes a
single ``\n`` only when the bar closes.  A line-based reader therefore sees
nothing until completion and then one blob holding every refresh — and the
UI's progress tap, matching the *first* ``n/total`` fragment in that blob,
shows the bar frozen at its initial frame (e.g. ``Decoding labeled cache:
0/1097``) until the work has already finished.

:func:`tqdm_logging_file` wraps the real stderr so each throttled refresh is
re-emitted as its own ``\n``-terminated line.  It writes straight to the
stream rather than through loguru on purpose: loguru's default subprocess
format would prepend a ``name:function:line - `` header that the progress tap
folds into the parsed ``desc``, polluting the status label with an internal
module path.  Writing the bare meter keeps ``desc`` clean while the parent's
own relay still timestamps the line for the session log.
"""

from __future__ import annotations

import sys
from typing import TextIO


class _NewlineTqdmFile:
    """File-like wrapper that turns tqdm's in-place redraws into full lines.

    Args:
        stream: Underlying text stream to write completed lines to, or
            ``None`` to resolve the live ``sys.stderr`` at each write.
    """

    def __init__(self, stream: TextIO | None) -> None:
        self._stream = stream

    def _resolve(self) -> TextIO:
        # Resolve sys.stderr lazily (not at construction) so a refresh tqdm
        # emits after a test's captured stderr is torn down — or any other
        # stream swap — lands on the live stream rather than a closed one.
        return self._stream if self._stream is not None else sys.stderr

    def write(self, message: str) -> None:
        """Write *message* as one newline-terminated line, dropping blanks.

        Args:
            message: A tqdm meter fragment, possibly carrying the carriage
                returns / whitespace of an in-place redraw.
        """
        text = message.strip("\r\n\t ")
        if text:
            self._resolve().write(text + "\n")

    def flush(self) -> None:
        """Flush the underlying stream (tqdm calls this after each refresh)."""
        self._resolve().flush()


def tqdm_logging_file(stream: TextIO | None = None) -> _NewlineTqdmFile:
    """Return a file-like object to pass as ``tqdm(file=...)``.

    Args:
        stream: Explicit stream to emit lines to.  When ``None`` (the
            default) the live ``sys.stderr`` is resolved at each write, so
            the meter follows stream swaps (e.g. pytest capture teardown)
            instead of holding a stale, possibly-closed reference.

    Returns:
        A wrapper that re-emits each tqdm refresh as a newline-terminated
        line so the subprocess relay and the UI's line-based reader see
        every update instead of a single frozen frame.
    """
    return _NewlineTqdmFile(stream)
