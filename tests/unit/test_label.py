#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the Label enum."""

from anomaly_match.datasets.Label import Label


class TestLabel:
    def test_label_values(self):
        assert Label.UNKNOWN == -1
        assert Label.NORMAL == 0
        assert Label.ANOMALY == 1

    def test_label_is_int(self):
        assert isinstance(Label.NORMAL, int)
        assert isinstance(Label.ANOMALY, int)
        assert isinstance(Label.UNKNOWN, int)

    def test_label_from_value(self):
        assert Label(-1) == Label.UNKNOWN
        assert Label(0) == Label.NORMAL
        assert Label(1) == Label.ANOMALY

    def test_label_members(self):
        members = list(Label)
        assert len(members) == 3
        assert Label.UNKNOWN in members
        assert Label.NORMAL in members
        assert Label.ANOMALY in members
