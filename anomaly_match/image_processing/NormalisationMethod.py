#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from enum import IntEnum


class NormalisationMethod(IntEnum):
    """Enum for normalisation methods."""

    CONVERSION_ONLY = 0
    LOG = 1
    ZSCALE = 2
    ASINH = 3

    @classmethod
    def get_dropdown_options(cls):
        """Returns a list of tuples (label, value) for use in dropdown widgets."""
        return [
            ("ConversionOnly", cls.CONVERSION_ONLY),
            ("LogStretch", cls.LOG),
            ("ZscaleInterval", cls.ZSCALE),
            ("Asinh", cls.ASINH),
        ]

    @classmethod
    def get_test_methods(cls):
        """Returns all methods for testing purposes."""
        return [cls.CONVERSION_ONLY, cls.LOG, cls.ZSCALE, cls.ASINH]
