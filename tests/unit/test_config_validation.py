#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import pytest
from loguru import logger

from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match.utils.validate_config import validate_config


@pytest.fixture
def caplog(caplog):
    """Configure loguru to use the caplog handler"""
    handler_id = logger.add(caplog.handler)
    yield caplog
    logger.remove(handler_id)


def test_default_cfg_validation(caplog):
    """Test that the default configuration passes validation without warnings."""
    # Get default config
    cfg = get_default_cfg()
    # image_size has no default - must be set by user
    cfg.normalisation.image_size = [224, 224]

    # Run validation
    validate_config(cfg)

    # Check logs for warnings
    assert "Found unexpected keys in config" not in caplog.text
    caplog.clear()
