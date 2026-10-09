#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""v2 benchmark runner: train once per label budget, score all images, evaluate.

This reproduces the AnomalyMatch paper's validation protocol under the v2
architecture. The paper refined a model across active-learning cycles; v2 drops
iterative refinement, so for each label budget we train a *single* model for a
fixed number of iterations and then score the entire dataset.

Pipeline per label budget:
  1. Build a fixed labeled set (``benchmark_datasets.build_labeled_csv``).
  2. Train a model via ``subprocess_scripts/training_process.py`` (subprocess).
  3. Score every image via ``predict_worker.py`` (subprocess) -> predictions.db.
  4. Evaluate AUROC / AUPRC / top-N precision on the full dataset, excluding the
     training-labeled images (matching the v1 methodology).
  5. Write per-run metrics + plots, then a cross-budget summary.

Runs one or more anomaly classes (``--anomaly-classes``), each swept over the
paper's label budgets, producing the paper's anomaly-detection-efficiency
figures per class and a cross-class comparison.

Run under the ``am`` conda environment so the spawned subprocesses inherit
torch + CUDA:

    conda run -n am python paper_scripts/v2/run_benchmark.py \
        --dataset miniimagenet --anomaly-classes all \
        --output-dir paper_scripts/v2/results/miniimagenet
"""

import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
from loguru import logger

# Make sibling modules importable when run as a script from any cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Repo root is three levels up: paper_scripts/v2/ -> paper_scripts/ -> repo.
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SUBPROCESS_SCRIPTS = os.path.join(_REPO_ROOT, "subprocess_scripts")

import benchmark_datasets as bd  # noqa: E402
import benchmark_plots as bp  # noqa: E402
import paper_plots as pp  # noqa: E402
from benchmark_config import (  # noqa: E402
    DEFAULT_LABEL_CONFIGS,
    BenchmarkParams,
    build_prediction_cfg,
    build_training_cfg,
)
from benchmark_metrics import evaluate_performance  # noqa: E402

from anomaly_match.prediction import AnomalyScoreDB  # noqa: E402

# Image extensions the AnomalyMatch image-folder source understands.
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".fits")


def _subprocess_env():
    """Return an environment with the repo + subprocess_scripts on PYTHONPATH."""
    env = dict(os.environ)
    extra = f"{_REPO_ROOT}:{_SUBPROCESS_SCRIPTS}"
    env["PYTHONPATH"] = f"{extra}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else extra
    return env


def _single_threaded_env(env):
    """Force per-process BLAS/OpenMP libraries to one thread.

    Prediction decodes images across several thread-pool workers. If each
    worker's numpy/fitsbolt call also spawns BLAS/OpenMP threads, the process
    oversubscribes the CPU (observed: 100+ threads thrashing, GPU idle). Pinning
    every math library to one thread lets the decode workers scale cleanly.

    Args:
        env: Base environment dict to extend.

    Returns:
        A copy of ``env`` with the math-library thread-count vars set to ``"1"``.
    """
    threaded = dict(env)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        threaded[var] = "1"
    return threaded


def list_image_files(image_dir):
    """Return sorted absolute paths of all images directly under ``image_dir``.

    Args:
        image_dir: Directory containing the dataset images (non-recursive).

    Returns:
        Sorted list of absolute image file paths.
    """
    files = []
    for name in os.listdir(image_dir):
        if name.lower().endswith(_IMAGE_EXTENSIONS):
            files.append(os.path.join(image_dir, name))
    return sorted(files)


def train_model(train_cfg_dict, labels_csv, run_dir):
    """Train a model in a subprocess and return the checkpoint path.

    Args:
        train_cfg_dict: Training config as a plain dict (``cfg.toDict()``).
        labels_csv: Path to the ``id,label`` labeled CSV.
        run_dir: Directory for the config pickle, progress log and checkpoint.

    Returns:
        Path to the saved ``.safetensors`` checkpoint.

    Raises:
        RuntimeError: If the training subprocess fails or does not report a
            terminal ``done`` status with a model path.
    """
    config_pkl = os.path.join(run_dir, "train_config.pkl")
    progress_file = os.path.join(run_dir, "train_progress.jsonl")
    with open(config_pkl, "wb") as f:
        pickle.dump(train_cfg_dict, f)
    # Fresh progress file so we never read stale lines from a previous attempt.
    open(progress_file, "w").close()

    cmd = [
        sys.executable,
        os.path.join(_SUBPROCESS_SCRIPTS, "training_process.py"),
        config_pkl,
        labels_csv,
        progress_file,
    ]
    logger.info(f"Launching training subprocess: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=_REPO_ROOT, env=_subprocess_env(), capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Training subprocess failed (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr[-4000:]}"
        )

    records = [json.loads(line) for line in open(progress_file) if line.strip()]
    done = [r for r in records if r.get("status") == "done"]
    if not done:
        raise RuntimeError(
            f"Training did not report 'done'. Last records: {records[-3:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
    model_path = done[-1]["model_path"]
    logger.info(f"Training complete. Checkpoint: {model_path}")
    return model_path


def predict_scores(pred_cfg_dict, image_files, pred_dir, max_workers):
    """Score all images in a subprocess and return (basename, score) pairs.

    Args:
        pred_cfg_dict: Prediction config as a plain dict.
        image_files: List of absolute image paths to score.
        pred_dir: Directory for the config pickle, file list and predictions.db.
        max_workers: Threads for image decode during prediction.

    Returns:
        List of ``(basename, score)`` tuples, score = softmax anomaly
        probability in [0, 1].

    Raises:
        RuntimeError: If the prediction subprocess fails or scores no images.
    """
    config_pkl = os.path.join(pred_dir, "predict_config.pkl")
    file_list = os.path.join(pred_dir, "image_files.txt")
    with open(config_pkl, "wb") as f:
        pickle.dump(pred_cfg_dict, f)
    with open(file_list, "w") as f:
        f.write("\n".join(image_files))

    cmd = [
        sys.executable,
        os.path.join(_THIS_DIR, "predict_worker.py"),
        config_pkl,
        file_list,
        "--top-n",
        str(len(image_files)),
        "--max-workers",
        str(max_workers),
    ]
    logger.info(f"Launching prediction subprocess over {len(image_files)} images")
    result = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        env=_single_threaded_env(_subprocess_env()),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Prediction subprocess failed (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr[-4000:]}"
        )

    # Read scores back from the pinned predictions DB (AnomalyScoreDB is
    # SQLite-only; no torch needed in this orchestrator process).
    db_path = os.path.join(pred_dir, "predictions.db")
    with AnomalyScoreDB(db_path) as db:
        n = db.get_count()
        if n == 0:
            raise RuntimeError(f"Prediction produced no scores in {db_path}")
        rows = db.get_results(sort_by="score_desc", limit=n, offset=0)
    # DB stores the full image path; ground truth is keyed on basename.
    pairs = [(os.path.basename(r["filename"]), r["score"]) for r in rows]
    logger.info(f"Read {len(pairs)} scores from {db_path}")
    return pairs


def _anomaly_prevalence(gt_df, anomaly_idx):
    """Return the anomaly fraction over the full dataset (for the perfect line)."""
    return float((gt_df["label_idx"] == anomaly_idx).mean())


def run_single_budget(
    dataset, anomaly_idx, anomaly_name, budget_idx, label_cfg, params, class_dir, gt_df
):
    """Run train -> predict -> evaluate for one label budget.

    Produces the paper figures (score histogram, ROC/PR, top-N anomaly-detection
    efficiency) via ``paper_plots``, which also pickles each plot's inputs under
    ``plots/plot_data/`` for later re-plotting, and saves the raw per-image
    scores so any figure can be regenerated without re-running prediction.

    Args:
        dataset: The :class:`benchmark_datasets.BenchmarkDataset`.
        anomaly_idx: Integer ``label_idx`` of the anomaly class.
        anomaly_name: Human-readable anomaly class name (for paths/titles).
        budget_idx: Index of this budget within the class sweep (0 = initial).
            Used as the "iteration"/cycle number in the paper plot functions.
        label_cfg: Dict with ``n_anomaly`` and ``n_nominal``.
        params: A :class:`BenchmarkParams`.
        class_dir: Output directory for this anomaly class.
        gt_df: Ground-truth DataFrame (``filename``, ``label_idx``).

    Returns:
        A tuple ``(summary_dict, (x, y))`` where ``(x, y)`` is the top-N
        anomaly-detection curve (percent inspected, percent anomalies found).
    """
    n_anomaly = label_cfg["n_anomaly"]
    n_nominal = label_cfg["n_nominal"]
    n_labeled = n_anomaly + n_nominal

    run_name = f"a{n_anomaly}_n{n_nominal}"
    run_dir = os.path.join(class_dir, run_name)
    pred_dir = os.path.join(run_dir, "prediction")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logger.info(
        f"===== {dataset.name}/{anomaly_name} budget {budget_idx}: "
        f"{n_anomaly} anomaly + {n_nominal} normal ====="
    )

    # 1. Labeled set
    labels_csv = os.path.join(run_dir, "labeled_data.csv")
    labeled = bd.build_labeled_csv(
        dataset, anomaly_idx, n_anomaly, n_nominal, labels_csv, params.seed
    )
    labeled_ids = set(labeled["id"])

    # 2. Train
    model_path = os.path.join(run_dir, "model.safetensors")
    train_cfg = build_training_cfg(dataset.image_dir, labels_csv, run_dir, model_path, params)
    model_path = train_model(train_cfg, labels_csv, run_dir)

    # 3. Predict over all images
    image_files = list_image_files(dataset.image_dir)
    pred_cfg = build_prediction_cfg(model_path, pred_dir, dataset.image_dir, params)
    pairs = predict_scores(pred_cfg, image_files, pred_dir, params.decode_workers)

    # 4. Evaluate on the unlabeled portion (exclude training-labeled images)
    eval_pairs = [(fn, s) for fn, s in pairs if fn not in labeled_ids]
    filenames = [fn for fn, _ in eval_pairs]
    scores = np.array([s for _, s in eval_pairs])
    metrics = evaluate_performance(scores, filenames, gt_df, anomaly_idx)

    logger.info(
        f"{dataset.name}/{anomaly_name} a{n_anomaly}: AUROC={metrics['auroc']:.4f} "
        f"AUPRC={metrics['auprc']:.4f} top-1% prec={metrics['top_1.0pct_precision']:.1f}% "
        f"top-1% recall={metrics['top_1.0pct_anomalies_found']:.1f}%"
    )

    # 5. Persist raw scores so every figure can be regenerated without re-running.
    merged = pd.merge(pd.DataFrame({"filename": filenames, "score": scores}), gt_df, on="filename")
    merged["true_anomaly"] = (merged["label_idx"] == anomaly_idx).astype(int)
    merged[["filename", "score", "label_idx", "true_anomaly"]].to_csv(
        os.path.join(run_dir, "scores.csv.gz"), index=False, compression="gzip"
    )

    # 6. Paper figures (each also pickles its inputs to plots/plot_data/).
    pp.plot_score_histogram(
        metrics["anomaly_scores"], metrics["normal_scores"], budget_idx, plots_dir
    )
    pp.plot_roc_prc_curves(metrics, budget_idx, plots_dir)
    x, y = pp.plot_top_n_anomaly_detection(
        scores, filenames, gt_df, anomaly_idx, budget_idx, plots_dir
    )

    summary = {
        "dataset": dataset.name,
        "anomaly_class": anomaly_name,
        "anomaly_idx": anomaly_idx,
        "budget_idx": budget_idx,
        "n_anomaly": n_anomaly,
        "n_nominal": n_nominal,
        "n_labeled": n_labeled,
        "num_train_iter": params.num_train_iter,
        "seed": params.seed,
        "n_evaluated": metrics["n_evaluated"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "top_0.1pct_anomalies_found": metrics["top_0.1pct_anomalies_found"],
        "top_0.1pct_precision": metrics["top_0.1pct_precision"],
        "top_1.0pct_anomalies_found": metrics["top_1.0pct_anomalies_found"],
        "top_1.0pct_precision": metrics["top_1.0pct_precision"],
    }
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    return summary, (x, y)


def run_single_class(dataset, anomaly_name, label_configs, params, output_dir, gt_df):
    """Run the full label-budget sweep for one anomaly class.

    Args:
        dataset: The :class:`benchmark_datasets.BenchmarkDataset`.
        anomaly_name: Human-readable anomaly class name.
        label_configs: List of ``{n_anomaly, n_nominal}`` budgets.
        params: A :class:`BenchmarkParams`.
        output_dir: Top-level results directory (one subdir per class).
        gt_df: Ground-truth DataFrame.

    Returns:
        A tuple ``(summaries, final_curve)`` where ``summaries`` is the list of
        per-budget summary dicts and ``final_curve`` is the ``(x, y)`` detection
        curve of the largest label budget (for the cross-class comparison).
    """
    anomaly_idx = bd.resolve_anomaly_class(dataset, anomaly_name)
    class_dir = os.path.join(output_dir, f"{dataset.name}_{anomaly_name}")
    class_plots = os.path.join(class_dir, "plots")
    os.makedirs(class_plots, exist_ok=True)
    prevalence = _anomaly_prevalence(gt_df, anomaly_idx)

    summaries = []
    detection_curves = {}
    for budget_idx, label_cfg in enumerate(label_configs):
        summary, curve = run_single_budget(
            dataset, anomaly_idx, anomaly_name, budget_idx, label_cfg, params, class_dir, gt_df
        )
        summaries.append(summary)
        detection_curves[budget_idx] = curve

    # Combined anomaly-detection efficiency across budgets (the headline figure).
    pp.plot_combined_anomaly_detection(detection_curves, class_plots, prevalence)

    class_summary = pd.DataFrame(summaries)
    class_summary.to_csv(os.path.join(class_dir, "summary.csv"), index=False)
    bp.plot_label_count_summary(
        class_summary,
        os.path.join(class_dir, "summary_vs_label_count.png"),
        f"{dataset.name} / {anomaly_name}: metrics vs label count",
    )
    logger.info(
        f"budget_idx -> labels for {anomaly_name}: "
        + ", ".join(f"{i}={c['n_anomaly']}a" for i, c in enumerate(label_configs))
    )
    return summaries, detection_curves[len(label_configs) - 1]


def resolve_classes(dataset, classes_arg):
    """Resolve the ``--anomaly-classes`` argument to a list of class names.

    Args:
        dataset: The :class:`benchmark_datasets.BenchmarkDataset`.
        classes_arg: ``"all"`` (every registered class) or a comma-separated list
            of class names / integer indices.

    Returns:
        List of anomaly class names.
    """
    if classes_arg == "all":
        return list(dataset.class_name_to_idx.keys())
    idx_to_name = {v: k for k, v in dataset.class_name_to_idx.items()}
    names = []
    for token in classes_arg.split(","):
        token = token.strip()
        idx = bd.resolve_anomaly_class(dataset, token)
        names.append(idx_to_name.get(idx, str(idx)))
    return names


def parse_args():
    """Parse command-line arguments for the benchmark runner.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description="AnomalyMatch v2 benchmark runner")
    parser.add_argument("--dataset", default="miniimagenet", choices=list(bd.REGISTRY.keys()))
    parser.add_argument(
        "--anomaly-classes",
        default="all",
        help="'all', or comma-separated class names / label_idx values",
    )
    parser.add_argument("--output-dir", default=os.path.join(_THIS_DIR, "results", "miniimagenet"))
    parser.add_argument("--num-train-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224, help="Square training resolution")
    parser.add_argument(
        "--label-configs",
        default=None,
        help="JSON list of {n_anomaly,n_nominal}; defaults to the paper trajectory",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Training DataLoader workers")
    parser.add_argument(
        "--decode-workers", type=int, default=8, help="Prediction image-decode threads"
    )
    return parser.parse_args()


def main():
    """Entry point: sweep label budgets for each requested anomaly class."""
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    dataset = bd.get_dataset(args.dataset)
    class_names = resolve_classes(dataset, args.anomaly_classes)

    label_configs = json.loads(args.label_configs) if args.label_configs else DEFAULT_LABEL_CONFIGS
    params = BenchmarkParams(
        num_train_iter=args.num_train_iter,
        image_size=(args.image_size, args.image_size),
        seed=args.seed,
        num_workers=args.num_workers,
        decode_workers=args.decode_workers,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, "benchmark.log"), level="INFO")
    logger.info(
        f"Benchmark: dataset={dataset.name} classes={class_names} "
        f"iters={params.num_train_iter} size={params.image_size} seed={params.seed} "
        f"budgets={label_configs}"
    )

    gt_df = bd.load_ground_truth(dataset)

    all_summaries = []
    final_curves = {}
    for anomaly_name in class_names:
        summaries, final_curve = run_single_class(
            dataset, anomaly_name, label_configs, params, args.output_dir, gt_df
        )
        all_summaries.extend(summaries)
        final_curves[anomaly_name] = final_curve
        # Write the master summary incrementally so partial results survive a crash.
        pd.DataFrame(all_summaries).to_csv(
            os.path.join(args.output_dir, "summary.csv"), index=False
        )

    # Cross-class anomaly-detection efficiency at the final (largest) budget.
    if len(final_curves) > 1:
        pp.plot_comparative_anomaly_detection(final_curves, args.output_dir)

    summary_df = pd.DataFrame(all_summaries)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    logger.info(f"Summary written to {os.path.join(args.output_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
