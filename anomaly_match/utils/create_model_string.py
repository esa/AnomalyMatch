#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import datetime


def create_model_string(cfg):
    """Creates a string from the arguments.

    Args:
        cfg (DotMap): config dictionary/dotmap

    Returns:
        str: string of the arguments
    """
    # fmt: off
    dir_name = (
        "model"
        + "_" + str(cfg.name)
        # Add date and time to the directory name
        + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    # fmt: on
    return dir_name
