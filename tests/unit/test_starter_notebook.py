#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[2] / "StarterNotebook.ipynb"


def test_starter_notebook_selects_am_kernel():
    """The notebook must ask for the 'am' kernel by name, not just display it.

    Jupyter matches kernels on ``kernelspec.name``; ``display_name`` is only a label.
    Re-saving the notebook from a kernel named ``python3`` silently opened it on base
    Python in the Datalab image, where AnomalyMatch is not installed.
    """
    kernelspec = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["kernelspec"]
    assert kernelspec["name"] == "am"
