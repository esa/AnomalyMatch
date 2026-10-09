#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the band-mixing split helpers in ``load_images``."""

import numpy as np
import pytest
from dotmap import DotMap

from anomaly_match.data_io.load_images import (
    fitsbolt_applies_channel_combination,
    fitsbolt_channel_combination,
)
from anomaly_match.utils.get_default_cfg import get_default_cfg

MATRIX = np.eye(3, 4)


def _cfg(fits_extension, channel_combination=MATRIX):
    cfg = get_default_cfg()
    cfg.normalisation.fits_extension = fits_extension
    cfg.normalisation.channel_combination = channel_combination
    return cfg


class TestFitsboltAppliesChannelCombination:
    @pytest.mark.parametrize("fits_extension", [0, "PRIMARY", [0, 1], ["VIS", "NIR-H"], (0,)])
    def test_fits_forms_answer_true(self, fits_extension):
        assert fitsbolt_applies_channel_combination(_cfg(fits_extension)) is True

    def test_unset_answers_false(self):
        assert fitsbolt_applies_channel_combination(_cfg(None)) is False

    @pytest.mark.parametrize("fits_extension", [DotMap(), 1.5, {"VIS": 0}])
    def test_unexpected_type_raises(self, fits_extension):
        """Guessing either way silently breaks band mixing, so fail hard instead.

        ``DotMap()`` is the ``cfg.copy()`` form of None (CLAUDE.md rule 13).
        """
        with pytest.raises(TypeError, match="fits_extension"):
            fitsbolt_applies_channel_combination(_cfg(fits_extension))


class TestFitsboltChannelCombination:
    def test_passes_matrix_through_for_fits(self):
        result = fitsbolt_channel_combination(_cfg([0, 1, 2, 3]))
        np.testing.assert_array_equal(result, MATRIX)

    def test_withholds_matrix_when_unset(self):
        assert fitsbolt_channel_combination(_cfg(None)) is None

    @pytest.mark.parametrize("matrix", [None, DotMap()])
    def test_non_matrix_becomes_none(self, matrix):
        assert fitsbolt_channel_combination(_cfg([0, 1], matrix)) is None

    def test_nested_list_is_coerced(self):
        result = fitsbolt_channel_combination(_cfg([0, 1], [[1.0, 0.0], [0.0, 1.0]]))
        np.testing.assert_array_equal(result, np.eye(2))
