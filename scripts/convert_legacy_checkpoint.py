#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Convert a trusted legacy AnomalyMatch checkpoint to safetensors."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anomaly_match.data_io.checkpoint_io import convert_legacy_checkpoint_to_safetensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_path", help="Trusted .pth or .pkl checkpoint")
    parser.add_argument("--output", help="Output path (the suffix is forced to .safetensors)")
    parser.add_argument(
        "--trusted",
        action="store_true",
        required=True,
        help="Confirm the checkpoint is trusted despite pickle's arbitrary-code risk",
    )
    args = parser.parse_args()

    output = convert_legacy_checkpoint_to_safetensors(
        args.legacy_path, args.output, trusted=args.trusted
    )
    print(output)


if __name__ == "__main__":
    main()
