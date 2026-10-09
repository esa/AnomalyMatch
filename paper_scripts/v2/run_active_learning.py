#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Active-learning benchmark: reproduce the paper's train -> relabel -> retrain loop.

This is the *fair* comparison to the original paper. Rather than sweeping static
label counts, it reproduces the paper's active-learning protocol per class:

  1. Start from the initial labeled set (5 anomalies + 495 nominal).
  2. Train a model for ``--iters-per-cycle`` iterations.
  3. Score every image; evaluate on the unlabeled portion.
  4. Relabel the top-K highest-scoring *true* anomalies (confirmed detections)
     and the top-K highest-scoring *true* nominals (false positives), exactly as
     the v1 ``find_mislabeled`` did, and append them to the labeled set.
  5. Repeat for ``--cycles`` cycles.

Cycle ``k`` here corresponds to v1 ``iteration_k``: cycle 1 is the cold start on
the initial 5/495 labels, and the final cycle matches v1's fully-refined model.

Differences from v1 (intentional, per project direction): each cycle retrains
*from the pretrained backbone* for ``--iters-per-cycle`` iterations rather than
continuing the previous cycle's weights for 100 iterations. Set
``--iters-per-cycle 100`` to match the paper's per-cycle budget.

Run under the ``am`` conda environment:

    conda run -n am python paper_scripts/v2/run_active_learning.py \
        --anomaly-classes all --iters-per-cycle 300 --cycles 3 \
        --output-dir paper_scripts/v2/results/miniimagenet_al
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from loguru import logger

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import benchmark_datasets as bd  # noqa: E402
import benchmark_plots as bp  # noqa: E402
import paper_plots as pp  # noqa: E402
from benchmark_config import BenchmarkParams, build_prediction_cfg, build_training_cfg  # noqa: E402
from benchmark_metrics import evaluate_performance  # noqa: E402
from run_benchmark import (  # noqa: E402
    _anomaly_prevalence,
    list_image_files,
    predict_scores,
    resolve_classes,
    train_model,
)

# Paper's initial labeled set for miniImageNet.
INITIAL_ANOMALY = 5
INITIAL_NOMINAL = 495


def select_new_labels(pairs, gt_df, anomaly_idx, labeled_ids, n_each):
    """Pick the top-scoring true anomalies and false positives to relabel.

    Mirrors the v1 ``find_mislabeled``: from the images not yet labeled, take the
    ``n_each`` highest-scoring images that are truly anomalies (confirmed
    detections) and the ``n_each`` highest-scoring images that are truly nominal
    (false positives to correct).

    Args:
        pairs: List of ``(basename, score)`` for all scored images.
        gt_df: Ground-truth DataFrame (``filename``, ``label_idx``).
        anomaly_idx: Integer ``label_idx`` of the anomaly class.
        labeled_ids: Set of already-labeled image basenames to exclude.
        n_each: Number of anomalies and of false positives to add.

    Returns:
        DataFrame with columns ``id`` and ``label`` for the new labels.
    """
    pred = pd.DataFrame({"filename": [f for f, _ in pairs], "score": [s for _, s in pairs]})
    pred = pred[~pred["filename"].isin(labeled_ids)]
    merged = pd.merge(pred, gt_df, on="filename")
    merged["true_anomaly"] = (merged["label_idx"] == anomaly_idx).astype(int)

    top_tp = merged[merged["true_anomaly"] == 1].nlargest(n_each, "score")
    top_fp = merged[merged["true_anomaly"] == 0].nlargest(n_each, "score")

    new = pd.concat(
        [
            pd.DataFrame({"id": top_tp["filename"], "label": "anomaly"}),
            pd.DataFrame({"id": top_fp["filename"], "label": "normal"}),
        ],
        ignore_index=True,
    )
    logger.info(f"Relabeled {len(top_tp)} confirmed anomalies + {len(top_fp)} false positives")
    return new


def run_active_learning_class(
    dataset, class_name, params, cycles, n_each, output_dir, gt_df, initial_anomaly, initial_nominal
):
    """Run the active-learning loop for one anomaly class.

    Args:
        dataset: The :class:`benchmark_datasets.BenchmarkDataset`.
        class_name: Anomaly class name.
        params: A :class:`BenchmarkParams` (``num_train_iter`` = iters per cycle).
        cycles: Number of active-learning cycles.
        n_each: Labels of each kind (anomaly / false positive) added per cycle.
        output_dir: Top-level results directory (one subdir per class).
        gt_df: Ground-truth DataFrame.
        initial_anomaly: Number of anomalies in the initial labelled set.
        initial_nominal: Number of nominals in the initial labelled set.

    Returns:
        Tuple ``(summaries, detection_curves)`` where ``detection_curves`` maps
        cycle number (1-based) to the ``(x, y)`` detection curve.
    """
    anomaly_idx = bd.resolve_anomaly_class(dataset, class_name)
    class_dir = os.path.join(output_dir, f"{dataset.name}_{class_name}")
    class_plots = os.path.join(class_dir, "plots")
    os.makedirs(class_plots, exist_ok=True)
    prevalence = _anomaly_prevalence(gt_df, anomaly_idx)
    image_files = list_image_files(dataset.image_dir)

    # Persistent labeled set, grown each cycle. Start from the paper's 5/495.
    labels_csv = os.path.join(class_dir, "labeled_data.csv")
    labeled = bd.build_labeled_csv(
        dataset, anomaly_idx, initial_anomaly, initial_nominal, labels_csv, params.seed
    )

    summaries = []
    detection_curves = {}
    for cycle in range(1, cycles + 1):
        cycle_dir = os.path.join(class_dir, f"cycle_{cycle}")
        pred_dir = os.path.join(cycle_dir, "prediction")
        plots_dir = os.path.join(cycle_dir, "plots")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        n_anom = int((labeled["label"] == "anomaly").sum())
        n_nom = int((labeled["label"] == "normal").sum())
        logger.info(
            f"===== {dataset.name}/{class_name} cycle {cycle}/{cycles}: "
            f"{n_anom} anomaly + {n_nom} normal ====="
        )

        # Train on the current labels (fresh from the pretrained backbone).
        model_path = os.path.join(cycle_dir, "model.safetensors")
        train_cfg = build_training_cfg(dataset.image_dir, labels_csv, cycle_dir, model_path, params)
        model_path = train_model(train_cfg, labels_csv, cycle_dir)

        # Score every image.
        pred_cfg = build_prediction_cfg(model_path, pred_dir, dataset.image_dir, params)
        pairs = predict_scores(pred_cfg, image_files, pred_dir, params.decode_workers)

        # Evaluate on the unlabeled portion (exclude the current labeled set).
        labeled_ids = set(labeled["id"])
        eval_pairs = [(fn, s) for fn, s in pairs if fn not in labeled_ids]
        filenames = [fn for fn, _ in eval_pairs]
        scores = np.array([s for _, s in eval_pairs])
        metrics = evaluate_performance(scores, filenames, gt_df, anomaly_idx)
        logger.info(
            f"{dataset.name}/{class_name} cycle {cycle}: AUROC={metrics['auroc']:.4f} "
            f"AUPRC={metrics['auprc']:.4f} top-1% prec={metrics['top_1.0pct_precision']:.1f}%"
        )

        # Persist raw scores + paper figures for this cycle.
        merged = pd.merge(
            pd.DataFrame({"filename": filenames, "score": scores}), gt_df, on="filename"
        )
        merged["true_anomaly"] = (merged["label_idx"] == anomaly_idx).astype(int)
        merged[["filename", "score", "label_idx", "true_anomaly"]].to_csv(
            os.path.join(cycle_dir, "scores.csv.gz"), index=False, compression="gzip"
        )
        pp.plot_score_histogram(
            metrics["anomaly_scores"], metrics["normal_scores"], cycle, plots_dir
        )
        pp.plot_roc_prc_curves(metrics, cycle, plots_dir)
        x, y = pp.plot_top_n_anomaly_detection(
            scores, filenames, gt_df, anomaly_idx, cycle, plots_dir
        )
        detection_curves[cycle] = (x, y)

        summaries.append(
            {
                "dataset": dataset.name,
                "anomaly_class": class_name,
                "anomaly_idx": anomaly_idx,
                "cycle": cycle,
                "n_anomaly": n_anom,
                "n_nominal": n_nom,
                "n_labeled": n_anom + n_nom,
                "iters_per_cycle": params.num_train_iter,
                "seed": params.seed,
                "n_evaluated": metrics["n_evaluated"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "top_0.1pct_anomalies_found": metrics["top_0.1pct_anomalies_found"],
                "top_0.1pct_precision": metrics["top_0.1pct_precision"],
                "top_1.0pct_anomalies_found": metrics["top_1.0pct_anomalies_found"],
                "top_1.0pct_precision": metrics["top_1.0pct_precision"],
            }
        )

        # Relabel for the next cycle (skip after the final evaluation).
        if cycle < cycles:
            new_labels = select_new_labels(pairs, gt_df, anomaly_idx, labeled_ids, n_each)
            labeled = pd.concat([labeled, new_labels], ignore_index=True).drop_duplicates("id")
            labeled.to_csv(labels_csv, index=False)

    pp.plot_combined_anomaly_detection(detection_curves, class_plots, prevalence)
    class_summary = pd.DataFrame(summaries)
    class_summary.to_csv(os.path.join(class_dir, "summary.csv"), index=False)
    bp.plot_label_count_summary(
        class_summary,
        os.path.join(class_dir, "summary_vs_label_count.png"),
        f"{dataset.name} / {class_name}: metrics vs active-learning cycle",
    )
    return summaries, detection_curves


def parse_args():
    """Parse command-line arguments.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description="AnomalyMatch v2 active-learning benchmark")
    parser.add_argument("--dataset", default="miniimagenet", choices=list(bd.REGISTRY.keys()))
    parser.add_argument(
        "--anomaly-classes",
        default="all",
        help="'all', or comma-separated class names / label_idx values",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(_THIS_DIR, "results", "miniimagenet_al")
    )
    parser.add_argument("--iters-per-cycle", type=int, default=300)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument(
        "--n-each", type=int, default=10, help="Anomalies and false positives added per cycle"
    )
    parser.add_argument(
        "--initial-anomaly",
        type=int,
        default=5,
        help="Initial labelled anomalies (paper: miniImageNet 5, GalaxyMNIST 10)",
    )
    parser.add_argument(
        "--initial-nominal",
        type=int,
        default=495,
        help="Initial labelled nominals (paper: miniImageNet 495, GalaxyMNIST 30)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=8)
    return parser.parse_args()


def main():
    """Run the active-learning loop for each requested anomaly class."""
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    dataset = bd.get_dataset(args.dataset)
    class_names = resolve_classes(dataset, args.anomaly_classes)
    params = BenchmarkParams(
        num_train_iter=args.iters_per_cycle,
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
        num_workers=args.num_workers,
        decode_workers=args.decode_workers,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, "active_learning.log"), level="INFO")
    logger.info(
        f"Active learning: dataset={dataset.name} classes={class_names} "
        f"cycles={args.cycles} iters/cycle={params.num_train_iter} n_each={args.n_each}"
    )

    gt_df = bd.load_ground_truth(dataset)

    all_summaries = []
    for class_name in class_names:
        summaries, _ = run_active_learning_class(
            dataset,
            class_name,
            params,
            args.cycles,
            args.n_each,
            args.output_dir,
            gt_df,
            args.initial_anomaly,
            args.initial_nominal,
        )
        all_summaries.extend(summaries)
        pd.DataFrame(all_summaries).to_csv(
            os.path.join(args.output_dir, "summary.csv"), index=False
        )

    logger.info(f"\n{pd.DataFrame(all_summaries).to_string(index=False)}")


if __name__ == "__main__":
    main()
