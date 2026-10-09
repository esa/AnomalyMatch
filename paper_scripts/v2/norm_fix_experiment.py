#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""One-cycle experiment: does matching timm's [-1,1] input range fix the gap?

Trains a single hourglass model on the paper's initial 5/495 labels (seed 42,
300 iters) and scores all images, exactly like ``run_active_learning`` cycle 1.
Run it once with ``transforms.py`` unpatched (reproduces AUROC ~0.814) and once
with a ``Normalize(0.5, 0.5)`` appended to the weak/strong/prediction transforms
(so inputs match ``tf_efficientnet_lite0.in1k``'s expected [-1,1]). A large jump
pins the regression on the missing input normalisation.

Run under the ``am`` env. Pass a tag to keep runs separate:
    python norm_fix_experiment.py <tag>
"""

import os
import sys

import numpy as np
import pandas as pd
from loguru import logger

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import benchmark_datasets as bd  # noqa: E402
from benchmark_config import BenchmarkParams, build_prediction_cfg, build_training_cfg  # noqa: E402
from benchmark_metrics import evaluate_performance  # noqa: E402
from run_benchmark import list_image_files, predict_scores, train_model  # noqa: E402

INITIAL_ANOMALY = 5
INITIAL_NOMINAL = 495


def main():
    """Train one 5/495 hourglass model, score everything, report AUROC."""
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    out_dir = os.path.join(_THIS_DIR, "results", "norm_fix", tag)
    os.makedirs(out_dir, exist_ok=True)

    dataset = bd.REGISTRY["miniimagenet"]
    anomaly_idx = bd.resolve_anomaly_class(dataset, "hourglass")
    gt_df = bd.load_ground_truth(dataset)
    image_files = list_image_files(dataset.image_dir)
    params = BenchmarkParams(num_train_iter=300, seed=42)

    labels_csv = os.path.join(out_dir, "labeled_data.csv")
    labeled = bd.build_labeled_csv(
        dataset, anomaly_idx, INITIAL_ANOMALY, INITIAL_NOMINAL, labels_csv, params.seed
    )

    model_path = os.path.join(out_dir, "model.safetensors")
    train_cfg = build_training_cfg(dataset.image_dir, labels_csv, out_dir, model_path, params)
    model_path = train_model(train_cfg, labels_csv, out_dir)

    pred_dir = os.path.join(out_dir, "prediction")
    os.makedirs(pred_dir, exist_ok=True)
    pred_cfg = build_prediction_cfg(model_path, pred_dir, dataset.image_dir, params)
    pairs = predict_scores(pred_cfg, image_files, pred_dir, params.decode_workers)

    labeled_ids = set(labeled["id"])
    eval_pairs = [(fn, s) for fn, s in pairs if fn not in labeled_ids]
    filenames = [fn for fn, _ in eval_pairs]
    scores = np.array([s for _, s in eval_pairs])
    metrics = evaluate_performance(scores, filenames, gt_df, anomaly_idx)
    logger.info(
        f"[{tag}] AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
        f"top-1%={metrics['top_1.0pct_precision']:.1f}%  (v1 iter1=0.9614, v2 baseline=0.8142)"
    )
    pd.DataFrame(
        [{"tag": tag, **{k: metrics[k] for k in ("auroc", "auprc", "top_1.0pct_precision")}}]
    ).to_csv(os.path.join(out_dir, "result.csv"), index=False)


if __name__ == "__main__":
    main()
