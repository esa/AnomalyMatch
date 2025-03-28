#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""Trivial test to see if model import still succeeds."""

import sys

sys.path.append("../..")


def test_import():
    import anomaly_match as am  # noqa: F401


if __name__ == "__main__":
    test_import()
