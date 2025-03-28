#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
from enum import IntEnum


class Label(IntEnum):
    """Enum for label values."""

    UNKNOWN = -1
    NORMAL = 0
    ANOMALY = 1
