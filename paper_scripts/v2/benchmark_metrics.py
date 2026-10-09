#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Evaluation metrics for the v2 benchmark.

These mirror the metrics reported in the AnomalyMatch paper so v2 results can be
compared directly against the published v1 numbers: AUROC, AUPRC, and the
fraction of anomalies recovered / precision within the top-scoring percentiles
of the ranked predictions.

The logic is ported from the v1 ``paper_utils.evaluate_performance`` and
``calculate_top_percentile_metrics`` so the definitions stay identical across
versions; only the plumbing that feeds in scores has changed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

# Percentiles (as percentages of the ranked pool) at which we report top-N
# detection. The paper highlights top-1% precision in particular.
DEFAULT_PERCENTILES = (0.1, 1.0)


def calculate_top_percentile_metrics(merged_df, percentiles=DEFAULT_PERCENTILES):
    """Compute anomaly recall and precision within top-scoring percentiles.

    Args:
        merged_df: DataFrame with a ``score`` column and a binary
            ``true_anomaly`` column (1 for anomaly, 0 otherwise).
        percentiles: Iterable of percentages (e.g. ``1.0`` for the top 1%).

    Returns:
        Dict mapping ``top_{p}pct_anomalies_found`` (percent of all anomalies
        recovered in the top p%) and ``top_{p}pct_precision`` (percent of the
        top p% that are anomalies) for each requested percentile.
    """
    sorted_df = merged_df.sort_values("score", ascending=False)
    total_anomalies = int(merged_df["true_anomaly"].sum())

    metrics = {}
    for percentile in percentiles:
        n_top = max(1, int(len(sorted_df) * percentile / 100))
        top = sorted_df.head(n_top)
        found = int(top["true_anomaly"].sum())

        pct_found = (found / total_anomalies) * 100 if total_anomalies > 0 else 0.0
        precision = (found / n_top) * 100 if n_top > 0 else 0.0
        metrics[f"top_{percentile:.1f}pct_anomalies_found"] = pct_found
        metrics[f"top_{percentile:.1f}pct_precision"] = precision

    return metrics


def evaluate_performance(scores, filenames, ground_truth_df, anomaly_class_idx):
    """Evaluate ranked predictions against ground truth.

    Args:
        scores: Array of anomaly scores (softmax anomaly probability in [0, 1]),
            one per predicted image.
        filenames: Sequence of filenames aligned with ``scores``.
        ground_truth_df: DataFrame with ``filename`` and ``label_idx`` columns
            covering every predicted image.
        anomaly_class_idx: Integer ``label_idx`` treated as the anomaly class.

    Returns:
        Dict of metrics: ``auroc``, ``auprc``, the raw ``precision``/``recall``
        arrays (for PR curves), the per-group ``anomaly_scores`` /
        ``normal_scores`` (for histograms), the number of evaluated images
        ``n_evaluated`` and anomaly count ``n_anomaly``, plus the top-percentile
        metrics from :func:`calculate_top_percentile_metrics`.

    Raises:
        ValueError: If any predicted filename is missing from the ground truth,
            which would silently drop rows and shrink the evaluation set — a
            mismatched filename convention rather than a legitimate evaluation.
    """
    pred_df = pd.DataFrame({"filename": list(filenames), "score": np.asarray(scores)})
    merged = pd.merge(pred_df, ground_truth_df, on="filename")
    # Require a lossless join: every scored file must have a ground-truth label.
    # A partial mismatch (some basenames absent from the labels CSV) would quietly
    # drop rows and change n_evaluated, so fail hard rather than under-count.
    if len(merged) != len(pred_df):
        raise ValueError(
            f"{len(pred_df) - len(merged)} of {len(pred_df)} predicted filenames are "
            "missing from the ground truth — check that prediction filenames match "
            "the labels CSV filename column"
        )

    merged["true_anomaly"] = (merged["label_idx"] == anomaly_class_idx).astype(int)
    y_true = merged["true_anomaly"].values
    y_score = merged["score"].values

    auroc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = auc(recall, precision)

    percentile_metrics = calculate_top_percentile_metrics(merged)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "anomaly_scores": merged[merged["true_anomaly"] == 1]["score"].values,
        "normal_scores": merged[merged["true_anomaly"] == 0]["score"].values,
        "n_evaluated": len(merged),
        "n_anomaly": int(y_true.sum()),
        **percentile_metrics,
    }
