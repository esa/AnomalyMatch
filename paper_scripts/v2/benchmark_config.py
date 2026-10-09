#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Configuration builders for the v2 benchmark.

Centralises the AnomalyMatch config used for a benchmark run so the training and
prediction phases stay consistent. The AnomalyMatch v2 defaults already match
the paper (EfficientNet-Lite0, SGD, lr 0.0075, weight decay 7.5e-4, momentum
0.9, batch size 16, uratio 5, p_cutoff 0.95, ema_m 0.99, num_train_iter 200), so
we only override paths, image size, iteration count and seed here.

The paper trained at 224x224. Using that size also selects the "hires"
unlabeled-pool cap (``unlabeled_pool_cap_hires`` = 10000, active above the
``unlabeled_pool_hires_threshold`` of 200 px), which reproduces the paper's
~10k-image unlabeled pool per training step.
"""

from dataclasses import dataclass

import anomaly_match as am

# Paper training resolution. Kept as a constant so training and any downstream
# assumptions share one source of truth (prediction reads size from the saved
# checkpoint, so it does not need this).
PAPER_IMAGE_SIZE = [224, 224]

# Paper active-learning label trajectory for miniImageNet. The paper starts at
# 5 anomalies + 495 nominal and, over 3 cycles, adds 10 anomalies + 10 nominal
# (high-scoring false positives) per cycle, ending at 35 anomalies + 525 nominal
# (i.e. nominal = 490 + n_anomaly; the nominal count barely grows). We approximate
# that trajectory with three *static* labeled sets — the initial, a midpoint, and
# the final — rather than running the active-learning loop. The final 35/525
# point is the direct comparison to the paper's headline miniImageNet numbers.
#
# Caveat: the paper's active learning *selects* the highest-scoring true
# anomalies / hardest false positives, whereas these static sets sample the
# added labels at random, so the final point may slightly under-perform the
# paper's actively-refined model at the same label count.
DEFAULT_LABEL_CONFIGS = [
    {"n_anomaly": 5, "n_nominal": 495},  # initial (paper cycle 0)
    {"n_anomaly": 20, "n_nominal": 510},  # in-between (~cycle 1.5)
    {"n_anomaly": 35, "n_nominal": 525},  # final (paper cycle 3, headline point)
]


@dataclass
class BenchmarkParams:
    """Tunable parameters for a benchmark run.

    Attributes:
        num_train_iter: Training iterations per model. The v2 pipeline trains a
            single model (no active-learning refinement), so this stands in for
            the paper's 3x100-iteration cycles.
        image_size: Training resolution as ``[width, height]``.
        seed: Random seed for labeled-set sampling and training.
        num_workers: DataLoader worker processes for training.
        prediction_batch_size: Images per prediction batch.
        decode_workers: Threads used to decode/preprocess images during
            prediction (raise above 1 to hide per-image NFS read latency).
    """

    num_train_iter: int = 200
    image_size: tuple = (224, 224)
    seed: int = 42
    num_workers: int = 4
    prediction_batch_size: int = 1000
    decode_workers: int = 8


def build_training_cfg(data_dir, label_file, output_dir, model_path, params):
    """Build the training config as a plain dict ready to pickle.

    Args:
        data_dir: Folder of images to train on (the full dataset image dir).
        label_file: Path to the ``id,label`` labeled CSV.
        output_dir: Run output directory (checkpoint + logs go here).
        model_path: Explicit path for the saved ``.safetensors`` checkpoint.
            The raw training subprocess does not create iteration directories,
            so the caller owns this path.
        params: A :class:`BenchmarkParams`.

    Returns:
        A ``dict`` (``cfg.toDict()``) suitable for ``pickle.dump`` and passing to
        ``subprocess_scripts/training_process.py``.
    """
    cfg = am.get_default_cfg()
    cfg.name = "v2_benchmark"
    cfg.data_dir = data_dir
    cfg.label_file = label_file
    cfg.output_dir = output_dir
    cfg.save_dir = output_dir
    cfg.model_path = model_path
    cfg.num_train_iter = params.num_train_iter
    cfg.seed = params.seed
    cfg.num_workers = params.num_workers
    cfg.normalisation.image_size = list(params.image_size)
    cfg.normalisation.n_output_channels = 3
    # No in-session test split; evaluation is done separately over the full set.
    cfg.test_ratio = 0.0
    return cfg.toDict()


def build_prediction_cfg(model_path, output_dir, data_dir, params):
    """Build the prediction config as a plain dict ready to pickle.

    The network, image size and normalisation are read back from the checkpoint
    by ``load_model``, so they are intentionally not set here.

    Args:
        model_path: Path to the trained ``.safetensors`` checkpoint.
        output_dir: Directory where ``predictions.db`` is written.
        data_dir: A real image directory. Set only to keep ``cfg.data_dir`` off
            the packaged test-data path, whose presence triggers a 64x64 /
            32-iteration override inside ``get_default_cfg``.
        params: A :class:`BenchmarkParams` (used for batch size and seed).

    Returns:
        A ``dict`` (``cfg.toDict()``) for the prediction worker.
    """
    cfg = am.get_default_cfg()
    cfg.name = "v2_benchmark_predict"
    cfg.data_dir = data_dir
    cfg.model_path = model_path
    # Pin a modest batch size. Left unset, ``estimate_batch_size`` picks a very
    # large batch for big GPUs, which collapses the whole run into ~1 batch that
    # decodes single-threaded before any DB write (GPU idle, no progress) and
    # reserves tens of GB. A fixed size gives incremental writes and steady
    # throughput.
    cfg.N_batch_prediction = params.prediction_batch_size
    cfg.output_dir = output_dir
    cfg.save_dir = output_dir
    # Pin the DB in output_dir so it is not relocated to local scratch on NFS.
    cfg.prediction_db_dir = output_dir
    # Score every image (including training-labeled ones); labeled images are
    # excluded later at evaluation time to match the v1 methodology.
    cfg.label_file = None
    cfg.seed = params.seed
    cfg.num_workers = params.num_workers
    return cfg.toDict()
