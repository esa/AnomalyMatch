#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
from dotmap import DotMap
import numpy as np
import ipywidgets as widgets


def print_cfg(cfg: DotMap):
    """Prints the config in a more structured and readable way.
    Args:
        cfg (DotMap): Config to print.
    """
    # Define groups of related parameters
    groups = {
        "General Settings": [
            "name",
            "log_level",
            "save_name",
            "save_dir",
            "data_dir",
            "output_dir",
            "label_file",
            "search_dir",
            "save_file",
        ],
        "Dataset Settings": [
            "seed",
            "test_ratio",
            "N_to_load",
            "size",
            "num_workers",
            "pin_memory",
            "prediction_file_type",
        ],
        "Model Settings": ["net", "pretrained", "num_classes", "num_channels"],
        "Training Settings": [
            "batch_size",
            "eval_batch_size",
            "num_train_iter",
            "num_eval_iter",
            "lr",
            "weight_decay",
            "opt",
            "momentum",
            "bn_momentum",
        ],
        "FixMatch Settings": [
            "ema_m",
            "hard_label",
            "temperature",
            "ulb_loss_ratio",
            "p_cutoff",
            "uratio",
            "oversample",
        ],
    }

    # Keep track of printed keys
    printed_keys = set()

    # Set formatting parameters
    section_width = 60
    key_width = 25

    # Print each group
    for group_name, keys in groups.items():
        # Print section header
        print(f"\n{'=' * section_width}")
        print(f"{group_name:^{section_width}}")  # Center the title
        print(f"{'-' * section_width}")

        group_keys = []

        # Collect available keys for this group
        for key in keys:
            if key in cfg:
                group_keys.append(key)
                printed_keys.add(key)

        # Print parameters in this group
        for key in group_keys:
            value = cfg[key]

            # Skip widget objects
            if isinstance(value, widgets.Widget):
                continue

            # Format the value based on its type
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                formatted_value = f"{value}"
            else:
                formatted_value = f"{value}"

            # Print with clear formatting
            print(f"{key:{key_width}}: {formatted_value}")

    # Print remaining parameters that weren't in any group
    remaining_keys = [
        k for k in cfg.keys() if k not in printed_keys and not isinstance(cfg[k], widgets.Widget)
    ]

    if remaining_keys:
        print(f"\n{'=' * section_width}")
        print(f"{'Other Parameters':^{section_width}}")
        print(f"{'-' * section_width}")

        for key in remaining_keys:
            value = cfg[key]

            # Format the value based on its type
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                formatted_value = f"{value}"
            else:
                formatted_value = f"{value}"

            # Print with clear formatting
            print(f"{key:{key_width}}: {formatted_value}")

    # Always print save_path at the end
    if "save_path" in cfg:
        print(f"\n{'=' * section_width}")
        print(f"{'Output Location':^{section_width}}")
        print(f"{'-' * section_width}")
        print(f"save_path{' ' * (key_width - 9)}: {cfg.save_path}")
