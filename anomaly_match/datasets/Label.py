#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Label enums for anomaly detection classification."""

from enum import IntEnum, StrEnum


class CsvLabel(StrEnum):
    """String labels used in labelled_data CSV files.

    Values match the literal strings stored in the ``label`` column.
    """

    ANOMALY = "anomaly"
    NORMAL = "normal"
    REMOVED = "removed"


#: Convenience aliases — avoids ``CsvLabel.ANOMALY`` at every call site.
LABEL_ANOMALY = CsvLabel.ANOMALY
LABEL_NORMAL = CsvLabel.NORMAL
LABEL_REMOVED = CsvLabel.REMOVED

#: Valid string labels in labelled_data CSV files.
VALID_CSV_LABELS: frozenset[str] = frozenset(CsvLabel)


class Label(IntEnum):
    """Numeric label enum for internal model use."""

    UNKNOWN = -1
    NORMAL = 0
    ANOMALY = 1
