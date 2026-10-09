#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Standalone prediction worker for the v2 benchmark.

Runs as its own subprocess so each benchmark iteration's GPU/CPU memory is fully
released when it exits — mirroring the v2 architecture where prediction is
subprocess-isolated. It loads a pickled prediction config and a newline-
delimited list of image paths, then scores them into ``predictions.db`` via the
production ``evaluate_files`` entry point.

Invoke with the repo root and ``subprocess_scripts/`` on ``PYTHONPATH`` so that
``prediction_utils`` and ``prediction_process`` import cleanly.
"""

import argparse

import torch

# ``prediction_utils`` and ``prediction_process`` live in subprocess_scripts/,
# which the launching orchestrator places on PYTHONPATH.
from prediction_process import evaluate_files  # noqa: E402

from prediction_utils import load_prediction_config  # noqa: E402


def main():
    """Parse arguments, load config + file list, and score all images."""
    parser = argparse.ArgumentParser(description="v2 benchmark prediction worker")
    parser.add_argument("config_path", help="Pickled prediction config (cfg.toDict())")
    parser.add_argument("file_list_path", help="Newline-delimited list of image paths")
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Threads for image decode/preprocess during prediction",
    )
    args = parser.parse_args()

    # load_prediction_config returns (cfg, batch_size). Because the benchmark
    # config pins N_batch_prediction, this batch_size is that fixed value.
    cfg, batch_size = load_prediction_config(args.config_path)

    # Keep torch's own intra-op pool single-threaded too; the decode workers
    # (``--max-workers``) provide the parallelism, and letting torch also spawn
    # a full thread pool per worker oversubscribes the CPU.
    torch.set_num_threads(1)

    with open(args.file_list_path) as f:
        files = [line.strip() for line in f if line.strip()]

    evaluate_files(
        files, cfg, top_n=args.top_n, batch_size=batch_size, max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
