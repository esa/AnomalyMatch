#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Compare AnomalyMatch v2 benchmark results against the original (v1) paper run.

The v1 paper results live at
``/media/team_workspaces/AnomalyMatch/paper_results/FullCorrectedOutput`` and
store, per class, the active-learning cycles (``iteration_0..3``) with pickled
anomaly-detection curves (``plot_data/data_for_combined_anomaly_detection.pkl``)
and a ``results_summary.csv``. Because the v2 harness reuses the *same* plotting
code, both sides' detection curves are computed identically and overlay directly.

For each class this writes, under
``/media/team_workspaces/AnomalyMatch/paper_results_v2/comparisons``:
  - ``<dataset>_<class>_detection_efficiency_v1_vs_v2.pdf`` — the headline overlay
    (v1 dashed vs v2 solid) for the cold-start and final label budgets.
  - ``<dataset>_metrics_v1_vs_v2.csv`` — AUROC/AUPRC/top-N, v1 vs v2, with deltas.
  - ``<dataset>_auroc_v1_vs_v2.pdf`` — grouped AUROC bars across classes.

Works for both ``miniimagenet`` and ``galaxymnist`` (``--dataset``); each dataset's
v1 runs and class→label_idx mapping are resolved automatically.

Label-budget correspondence used for the overlay:
  - cold start: v1 iteration 1 (5 anomalies, before active learning)
    <-> v2 budget 0 (5 anomalies).
  - final:      v1 iteration 3 (after 3 AL cycles, ~25 anomalies)
    <-> v2 budget 2 (35 anomalies).
"""

import argparse
import glob
import os
import pickle
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import matplotlib

matplotlib.use("Agg")
import benchmark_datasets as bd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

# Where the v1 paper runs live per dataset. Each dataset stores its classes under
# ``<root>/<class_glob>`` (formatted with the anomaly label_idx). The nominal
# ratio differs by dataset (miniImageNet 1%, GalaxyMNIST 25%) but the AL protocol
# is the same, so the detection-curve comparison is apples-to-apples.
_V1_BASE = "/media/team_workspaces/AnomalyMatch/paper_results/FullCorrectedOutput"
V1_LAYOUTS = {
    "miniimagenet": {
        "root": f"{_V1_BASE}/miniimagenet",
        "class_glob": "mini_class{idx}_ratio0.01/miniimagenet_anomaly{idx}_*",
    },
    "galaxymnist": {
        "root": f"{_V1_BASE}/galaxymnist",
        "class_glob": "galaxy_class{idx}_ratio0.25/galaxymnist_anomaly{idx}_*",
    },
}

# v1 active-learning iterations to compare against: iteration 1 = first trained
# model on the initial 5/495 labels (cold start); iteration 3 = after all three
# AL cycles (final). The v2 side uses its first/last step (budget 0 / cycle 1 =
# cold; last budget / cycle 3 = final), derived per class.
COLD_V1_ITER = 1
FINAL_V1_ITER = 3

_BLUE = "#1f77b4"
_RED = "#d62728"
_PERFECT = "#2ca02c"


def load_v1_class(dataset, anomaly_idx):
    """Load v1 detection curves + summary for one class.

    Args:
        dataset: Dataset name (``"miniimagenet"`` or ``"galaxymnist"``).
        anomaly_idx: The anomaly ``label_idx`` for this class.

    Returns:
        Dict with ``curves`` ({iteration: (x, y)}), ``prevalence`` (float), and
        ``summary`` (the ``results_summary.csv`` row as a Series), or ``None`` if
        the class directory is missing.
    """
    layout = V1_LAYOUTS[dataset]
    matches = glob.glob(os.path.join(layout["root"], layout["class_glob"].format(idx=anomaly_idx)))
    if not matches:
        return None
    base = matches[0]
    combined = os.path.join(base, "plots", "plot_data", "data_for_combined_anomaly_detection.pkl")
    with open(combined, "rb") as f:
        data = pickle.load(f)
    summary = pd.read_csv(os.path.join(base, "results_summary.csv")).iloc[0]
    return {
        "curves": data["detection_curves"],
        "prevalence": data["anomaly_prevalence"],
        "summary": summary,
    }


def load_v2_class(v2_dir, dataset, class_name):
    """Load v2 detection curves + per-budget summary for one class.

    Args:
        v2_dir: The v2 run output directory (contains ``<dataset>_<class>/``).
        dataset: Dataset name (e.g. ``miniimagenet``).
        class_name: Anomaly class name.

    Returns:
        Dict with ``curves`` ({budget_idx: (x, y)}) and ``summary`` (DataFrame of
        per-budget rows), or ``None`` if the class has not finished yet.
    """
    class_dir = os.path.join(v2_dir, f"{dataset}_{class_name}")
    combined = os.path.join(
        class_dir, "plots", "plot_data", "data_for_combined_anomaly_detection.pkl"
    )
    summary_csv = os.path.join(class_dir, "summary.csv")
    if not (os.path.exists(combined) and os.path.exists(summary_csv)):
        return None
    with open(combined, "rb") as f:
        data = pickle.load(f)
    summary = pd.read_csv(summary_csv)
    # A budget-sweep run indexes rows by ``budget_idx`` (curves keyed 0..2); an
    # active-learning run indexes by ``cycle`` (curves keyed 1..3). Detect which.
    index_col = "cycle" if "cycle" in summary.columns else "budget_idx"
    return {
        "curves": data["detection_curves"],
        "summary": summary,
        "index_col": index_col,
    }


def _perfect_curve(prevalence):
    """Return (x, y) percentages for the perfect-detection reference line."""
    x = np.unique(
        np.concatenate(
            [np.logspace(np.log10(0.0001), np.log10(0.1), 500), np.linspace(0.1, 100, 500)]
        )
    )
    y = np.minimum(x / prevalence, 100.0) if prevalence > 0 else x
    return x, y


def plot_class_comparison(class_name, v1, v2, out_path):
    """Overlay v1 vs v2 detection-efficiency curves for one class.

    Args:
        class_name: Anomaly class name (for the title).
        v1: The dict from :func:`load_v1_class`.
        v2: The dict from :func:`load_v2_class`.
        out_path: Output PDF path.
    """
    v2_summary = v2["summary"].set_index(v2["index_col"])
    v1_s = v1["summary"]
    # First and last v2 step (budget 0 / cycle 1 = cold; last = final).
    cold_key = min(v2["curves"].keys())
    final_key = max(v2["curves"].keys())

    fig, ax = plt.subplots(figsize=(8, 8))

    # Cold start (5 anomalies).
    x1, y1 = v1["curves"][COLD_V1_ITER]
    x2, y2 = v2["curves"][cold_key]
    v2_cold_auroc = v2_summary.loc[cold_key, "auroc"]
    ax.plot(
        x1,
        y1,
        color=_BLUE,
        ls="--",
        lw=2.5,
        label=f"v1 cold (5 anom), AUROC={v1_s['first_iter_auroc']:.3f}",
    )
    ax.plot(
        x2, y2, color=_BLUE, ls="-", lw=2.5, label=f"v2 cold (5 anom), AUROC={v2_cold_auroc:.3f}"
    )

    # Final budget.
    x1f, y1f = v1["curves"][FINAL_V1_ITER]
    x2f, y2f = v2["curves"][final_key]
    v2_final_auroc = v2_summary.loc[final_key, "auroc"]
    v2_final_anom = int(v2_summary.loc[final_key, "n_anomaly"])
    ax.plot(
        x1f,
        y1f,
        color=_RED,
        ls="--",
        lw=2.5,
        label=f"v1 final (AL, ~25 anom), AUROC={v1_s['final_auroc']:.3f}",
    )
    ax.plot(
        x2f,
        y2f,
        color=_RED,
        ls="-",
        lw=2.5,
        label=f"v2 final ({v2_final_anom} anom), AUROC={v2_final_auroc:.3f}",
    )

    xp, yp = _perfect_curve(v1["prevalence"])
    ax.plot(xp, yp, color=_PERFECT, ls=":", lw=2, alpha=0.8, label="Perfect detection")

    for xv in (0.1, 1):
        ax.axvline(x=xv, color="black", ls=":", alpha=0.4)

    ax.set_xscale("log")
    ax.set_xlim(0.008, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.set_xlabel("% of top-scoring predictions inspected")
    ax.set_ylabel("% of total anomalies found")
    ax.set_title(f"miniImageNet / {class_name}: v2 (solid) vs v1 (dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def build_metrics_table(rows, out_path):
    """Write the v1-vs-v2 metrics comparison CSV.

    Args:
        rows: List of per-class metric dicts.
        out_path: Output CSV path.

    Returns:
        The metrics DataFrame.
    """
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}")
    return df


def plot_auroc_bars(metrics_df, out_path):
    """Grouped AUROC bars (v1 vs v2, cold + final) across classes.

    Args:
        metrics_df: The metrics DataFrame from :func:`build_metrics_table`.
        out_path: Output PDF path.
    """
    classes = metrics_df["class"].tolist()
    x = np.arange(len(classes))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * w, metrics_df["v1_cold_auroc"], w, label="v1 cold", color=_BLUE, alpha=0.5)
    ax.bar(x - 0.5 * w, metrics_df["v2_cold_auroc"], w, label="v2 cold", color=_BLUE)
    ax.bar(x + 0.5 * w, metrics_df["v1_final_auroc"], w, label="v1 final", color=_RED, alpha=0.5)
    ax.bar(x + 1.5 * w, metrics_df["v2_final_auroc"], w, label="v2 final", color=_RED)
    ax.set_xticks(x, classes)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("miniImageNet AUROC: v1 vs v2 (cold start and final budget)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


def parse_args():
    """Parse command-line arguments.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Compare v2 vs v1 benchmark results")
    parser.add_argument(
        "--v2-dir",
        default=os.path.join(here, "results", "miniimagenet"),
        help="v2 run output directory",
    )
    parser.add_argument("--dataset", default="miniimagenet")
    parser.add_argument(
        "--out-dir",
        default="/media/team_workspaces/AnomalyMatch/paper_results_v2/comparisons",
    )
    return parser.parse_args()


def main():
    """Build all v1-vs-v2 comparison figures and the metrics table."""
    args = parse_args()
    logger.remove()
    logger.add(lambda m: print(m, end=""))
    os.makedirs(args.out_dir, exist_ok=True)

    class_to_idx = bd.get_dataset(args.dataset).class_name_to_idx
    metric_rows = []
    for class_name, anomaly_idx in class_to_idx.items():
        v1 = load_v1_class(args.dataset, anomaly_idx)
        v2 = load_v2_class(args.v2_dir, args.dataset, class_name)
        if v1 is None:
            logger.warning(f"No v1 results for {class_name}; skipping")
            continue
        if v2 is None:
            logger.info(f"v2 for {class_name} not finished yet; skipping")
            continue

        plot_class_comparison(
            class_name,
            v1,
            v2,
            os.path.join(
                args.out_dir, f"{args.dataset}_{class_name}_detection_efficiency_v1_vs_v2.pdf"
            ),
        )

        v2s = v2["summary"].set_index(v2["index_col"])
        cold_key = min(v2["curves"].keys())
        final_key = max(v2["curves"].keys())
        metric_rows.append(
            {
                "class": class_name,
                "v1_cold_auroc": round(v1["summary"]["first_iter_auroc"], 4),
                "v2_cold_auroc": round(v2s.loc[cold_key, "auroc"], 4),
                "v1_final_auroc": round(v1["summary"]["final_auroc"], 4),
                "v2_final_auroc": round(v2s.loc[final_key, "auroc"], 4),
                "v1_final_auprc": round(v1["summary"]["final_auprc"], 4),
                "v2_final_auprc": round(v2s.loc[final_key, "auprc"], 4),
                "v1_final_top1pct_prec": round(v1["summary"]["top_1.0pct_precision"], 2),
                "v2_final_top1pct_prec": round(v2s.loc[final_key, "top_1.0pct_precision"], 2),
                "delta_final_auroc": round(
                    v2s.loc[final_key, "auroc"] - v1["summary"]["final_auroc"], 4
                ),
            }
        )

    if not metric_rows:
        logger.warning("No classes had both v1 and v2 results; nothing to compare yet.")
        return

    metrics_df = build_metrics_table(
        metric_rows, os.path.join(args.out_dir, f"{args.dataset}_metrics_v1_vs_v2.csv")
    )
    plot_auroc_bars(metrics_df, os.path.join(args.out_dir, f"{args.dataset}_auroc_v1_vs_v2.pdf"))
    logger.info("\n" + metrics_df.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
