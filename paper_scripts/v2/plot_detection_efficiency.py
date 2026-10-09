#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Anomaly-detection-efficiency figure: all classes x all AL cycles for a dataset.

For each class, overlays every active-learning cycle's detection curve (% of all
anomalies found vs % of the dataset inspected, top-scored first) in one panel,
annotating the absolute "found N / M anomalies" at the 1%-inspected point. Reads
the per-cycle ``scores.csv.gz`` written by ``run_active_learning.py``.

Example::

    python plot_detection_efficiency.py --dataset galaxymnist \
        --v2-dir results/miniimagenet_al/galaxymnist --out-dir comparisons
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import benchmark_datasets as bd  # noqa: E402

_CYCLE_COLORS = {1: "#4C9BE8", 2: "#E8A33D", 3: "#2CA02C"}


def _detection_curve(scores_csv):
    """Return (x=%inspected, y=%found, total_anomalies, cum_found, auroc) or None."""
    if not os.path.exists(scores_csv):
        return None
    df = pd.read_csv(scores_csv).sort_values("score", ascending=False).reset_index(drop=True)
    total = int(df["true_anomaly"].sum())
    cum = df["true_anomaly"].cumsum().to_numpy()
    n = len(df)
    x = np.arange(1, n + 1) / n * 100.0
    y = cum / total * 100.0
    auroc = roc_auc_score(df["true_anomaly"], df["score"])
    return x, y, total, cum, auroc


def plot_dataset(dataset, v2_dir, out_dir, cycles):
    """Write the all-cycles detection-efficiency grid for one dataset.

    Args:
        dataset: Dataset name (``"miniimagenet"`` or ``"galaxymnist"``).
        v2_dir: Directory containing ``<dataset>_<class>/cycle_<k>/scores.csv.gz``.
        out_dir: Directory to write the figure into.
        cycles: Number of active-learning cycles to overlay.

    Returns:
        The output ``.png`` path, or ``None`` if no class had any scores.
    """
    classes = list(bd.get_dataset(dataset).class_name_to_idx)
    ncol = 3 if len(classes) > 4 else 2
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow), squeeze=False)
    any_data = False
    for idx, cls in enumerate(classes):
        ax = axes[idx // ncol][idx % ncol]
        total = n = None
        for cyc in range(1, cycles + 1):
            curve = _detection_curve(
                os.path.join(v2_dir, f"{dataset}_{cls}", f"cycle_{cyc}", "scores.csv.gz")
            )
            if curve is None:
                continue
            any_data = True
            x, y, total, cum, auroc = curve
            n = len(x)
            found = int(cum[max(0, int(round(0.01 * n)) - 1)])
            ax.plot(
                x,
                y,
                color=_CYCLE_COLORS.get(cyc, None),
                lw=2.2,
                label=f"Cycle {cyc} (AUROC {auroc:.3f}) — top 1%: {found}/{total}",
            )
        if total:
            prevalence = total / n
            xp = np.logspace(-2, 2, 500)
            ax.plot(xp, np.minimum(xp / prevalence, 100), "k--", alpha=0.4, label="Perfect")
            ax.axvline(1.0, color="grey", ls=":", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlim(0.01, 100)
        ax.set_ylim(0, 102)
        ax.set_xlabel("% of dataset inspected (top-scored first)")
        ax.set_ylabel("% of anomalies found")
        ax.set_title(f"{cls}  (n_anomalies = {total})", fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=7.5, loc="lower right")
    for j in range(len(classes), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if not any_data:
        plt.close(fig)
        return None
    fig.suptitle(
        f"Anomaly detection efficiency — {dataset} (all AL cycles)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{dataset}_detection_efficiency_allcycles.png")
    for path in (png, png[:-4] + ".pdf"):
        fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return png


def parse_args():
    """Parse command-line arguments.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="miniimagenet", choices=list(bd.REGISTRY.keys()))
    p.add_argument(
        "--v2-dir",
        required=True,
        help="Directory with <dataset>_<class>/cycle_<k>/scores.csv.gz",
    )
    p.add_argument(
        "--out-dir",
        default="/media/team_workspaces/AnomalyMatch/paper_results_v2/comparisons",
    )
    p.add_argument("--cycles", type=int, default=3)
    return p.parse_args()


def main():
    """Build the all-cycles detection-efficiency figure for one dataset."""
    args = parse_args()
    out = plot_dataset(args.dataset, args.v2_dir, args.out_dir, args.cycles)
    print(f"wrote {out}" if out else "no scores found; nothing plotted")


if __name__ == "__main__":
    main()
