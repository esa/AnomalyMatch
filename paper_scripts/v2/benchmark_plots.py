#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Lean, dependency-light plotting helpers for the v2 benchmark.

Provides self-contained implementations of the score histogram, ROC + PR curves,
top-N anomaly-detection curve, and the metric-vs-label-count summary. The
active-learning runner wires in :func:`plot_label_count_summary` for the
per-class summary figure; the per-cycle score/ROC/top-N figures in a run are
produced by the vendored :mod:`paper_plots` (so they match the v1 paper style
exactly). The other functions here are kept as a lightweight, standalone
alternative for ad-hoc plotting without the ~2000-line ``paper_plots.py``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


def plot_score_histogram(anomaly_scores, normal_scores, out_path, title):
    """Plot overlaid histograms of anomaly vs normal scores.

    Args:
        anomaly_scores: Scores for ground-truth anomalies.
        normal_scores: Scores for ground-truth normal images.
        out_path: Output image path.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(
        normal_scores,
        bins=bins,
        alpha=0.6,
        label=f"normal (n={len(normal_scores)})",
        color="#4c72b0",
        density=True,
    )
    ax.hist(
        anomaly_scores,
        bins=bins,
        alpha=0.6,
        label=f"anomaly (n={len(anomaly_scores)})",
        color="#c44e52",
        density=True,
    )
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_prc(metrics, out_path, title):
    """Plot ROC and precision-recall curves side by side.

    Args:
        metrics: Metrics dict from ``evaluate_performance`` (uses
            ``anomaly_scores``, ``normal_scores``, ``precision``, ``recall``,
            ``auroc``, ``auprc``).
        out_path: Output image path.
        title: Figure suptitle.
    """
    y_score = np.concatenate([metrics["anomaly_scores"], metrics["normal_scores"]])
    y_true = np.concatenate(
        [np.ones(len(metrics["anomaly_scores"])), np.zeros(len(metrics["normal_scores"]))]
    )
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, (ax_roc, ax_prc) = plt.subplots(1, 2, figsize=(12, 5))
    ax_roc.plot(fpr, tpr, color="#4c72b0", label=f"AUROC = {metrics['auroc']:.4f}")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_xlabel("false positive rate")
    ax_roc.set_ylabel("true positive rate")
    ax_roc.set_title("ROC")
    ax_roc.legend(loc="lower right")

    ax_prc.plot(
        metrics["recall"],
        metrics["precision"],
        color="#c44e52",
        label=f"AUPRC = {metrics['auprc']:.4f}",
    )
    ax_prc.set_xlabel("recall")
    ax_prc.set_ylabel("precision")
    ax_prc.set_title("Precision-Recall")
    ax_prc.legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_n_detection(scores, y_true, out_path, title):
    """Plot cumulative anomaly recovery as a function of images inspected.

    Args:
        scores: Anomaly scores for the evaluated pool.
        y_true: Binary ground-truth (1 = anomaly) aligned with ``scores``.
        out_path: Output image path.
        title: Figure title.
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    order = np.argsort(-scores)
    ranked = y_true[order]
    cumulative = np.cumsum(ranked)
    total_anomalies = cumulative[-1]
    n = len(scores)

    inspected_frac = np.arange(1, n + 1) / n
    recovered_frac = cumulative / total_anomalies
    prevalence = total_anomalies / n

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(inspected_frac, recovered_frac, color="#4c72b0", label="model")
    # Perfect: every top-scoring image is an anomaly until they run out.
    perfect_x = np.linspace(0, 1, 200)
    perfect_y = np.minimum(perfect_x / prevalence, 1.0)
    ax.plot(perfect_x, perfect_y, "g--", alpha=0.6, label="perfect")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.4, label="random")
    ax.set_xlabel("fraction of images inspected")
    ax.set_ylabel("fraction of anomalies recovered")
    ax.set_xscale("log")
    ax.set_xlim(1 / n, 1)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_label_count_summary(summary_df, out_path, title):
    """Plot AUROC / AUPRC / top-1% precision against total labeled count.

    Args:
        summary_df: DataFrame with columns ``n_labeled``, ``auroc``, ``auprc``,
            and ``top_1.0pct_precision``.
        out_path: Output image path.
        title: Figure title.
    """
    df = summary_df.sort_values("n_labeled")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["n_labeled"], df["auroc"], "o-", label="AUROC", color="#4c72b0")
    ax.plot(df["n_labeled"], df["auprc"], "s-", label="AUPRC", color="#c44e52")
    ax.plot(
        df["n_labeled"],
        df["top_1.0pct_precision"] / 100.0,
        "^-",
        label="top-1% precision",
        color="#55a868",
    )
    ax.set_xlabel("total labeled images")
    ax.set_ylabel("score")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
