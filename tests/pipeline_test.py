#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
"""Testing a mininal pipeline"""

import pytest
import ipywidgets as widgets
import anomaly_match as am


@pytest.fixture(scope="module")
def pipeline_config():
    out = widgets.Output(
        layout=widgets.Layout(
            border="1px solid white", height="400px", background_color="black", overflow="auto"
        ),
        style={"color": "white"},
    )
    progress_bar = widgets.FloatProgress(
        value=0.0,
        min=0.0,
        max=1.0,
    )

    cfg = am.get_default_cfg()
    am.set_log_level("trace", cfg)
    cfg.data_dir = "tests/test_data/"
    cfg.size = [64, 64]
    cfg.num_train_iter = 10
    cfg.progress_bar = progress_bar
    return cfg, out


def test_pipeline(pipeline_config):
    cfg, out = pipeline_config
    session = am.Session(cfg)
    session.out = out
    session.train(cfg)


if __name__ == "__main__":
    test_pipeline()
