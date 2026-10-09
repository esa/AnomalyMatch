[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# AnomalyMatch v2 benchmark validation

Validation harness that reproduces the AnomalyMatch paper
([arXiv:2505.03509](https://arxiv.org/abs/2505.03509)) benchmark methodology
against the **v2** architecture, to check that the deep changes between `main`
and the v2 line did **not** cause performance regressions.

## What changed vs the v1 paper benchmark

The original `paper_scripts/paper_benchmark.py` drove the old in-process
`Session` API (`session.train(...)`, `session.model.eval_model`,
`session.label_image(...)`). That entire API was removed on the v2 line
(commit `0d44c3fd — "remove paper_scripts/ (incompatible with new Session
API)"`). v2 runs training and prediction as **isolated subprocesses**. This
harness is a clean rewrite against that architecture.

The other deliberate change (per project direction): **no active-learning
refinement**. v1 refined a model across 3×100-iteration cycles, relabelling the
top false-positives/true-positives between cycles. v2 has dropped iterative
refinement, so here we train a **single** model for a fixed number of iterations
(~200) per label budget and evaluate once.

## Files

| File | Purpose |
|------|---------|
| `benchmark_datasets.py` | Dataset registry + fixed labeled-set construction (`id,label` CSV). API-independent. |
| `benchmark_metrics.py`  | AUROC / AUPRC / top-N precision, ported verbatim from v1 so numbers are comparable. |
| `benchmark_config.py`   | Builds the training + prediction configs (v2 defaults already match the paper). |
| `predict_worker.py`     | Standalone prediction subprocess wrapping the production `evaluate_files`. |
| `paper_plots.py`        | The **v1 paper plotting code**, reused as-is: score histogram, ROC/PR, and the top-N anomaly-detection efficiency figures (per-budget + combined + cross-class). Only the two package-relative imports were fixed and `save_plot_data` inlined. |
| `plot_colors.py`        | Paper colour scheme (dependency of `paper_plots.py`). |
| `benchmark_plots.py`    | One extra, non-paper figure: metric-vs-label-count summary. |
| `run_benchmark.py`      | Orchestrator: for each class × label budget → train → score all images → evaluate → paper plots. |
| `_reference_v1/`        | The original v1 paper scripts (recovered from git history) for reference. |

## Anomaly class indices

The miniImageNet labels CSV stores only integer labels, so class indices were
verified by rendering sample images. The five paper anomaly classes:

| class | `label_idx` | prevalence |
|-------|-------------|------------|
| hourglass (primary) | 57 | 1.0% (650 / 65000) |
| piano   | 85 | 1.0% |
| guitar  | 48 | 1.0% |
| printer (photocopier) | 68 | 1.0% |
| orange  | 95 | 1.0% |

GalaxyMNIST uses readable class names (`smooth_round`, `smooth_cigar`,
`edge_on_disk`, `unbarred_spiral`).

## Label budgets (paper active-learning trajectory)

The paper starts at **5 anomaly / 495 nominal** and, over 3 active-learning
cycles, adds **10 anomaly + 10 nominal** per cycle, ending at **35 anomaly /
525 nominal** (nominal ≈ 490 + n_anomaly; the nominal count barely grows). We
approximate that trajectory with three *static* labeled sets rather than running
the AL loop:

| budget | n_anomaly | n_nominal | ≈ paper stage |
|--------|-----------|-----------|---------------|
| initial | 5 | 495 | cycle 0 (cold start) |
| in-between | 20 | 510 | ~cycle 1.5 |
| final | 35 | 525 | cycle 3 — **direct comparison to the paper's headline** |

Caveat: the paper's AL *selects* the highest-scoring true anomalies / hardest
false positives, whereas these static sets add labels at random, so the final
point may slightly under-perform the paper's actively-refined model.

## Running

Must run under the `am` conda env so the training/prediction subprocesses get
torch + CUDA:

```bash
# All five miniImageNet classes, 3 budgets each, 200 iters
conda run -n am python paper_scripts/v2/run_benchmark.py \
    --dataset miniimagenet --anomaly-classes all \
    --num-train-iter 200 \
    --output-dir paper_scripts/v2/results/miniimagenet

# A single class (or a comma-separated subset)
conda run -n am python paper_scripts/v2/run_benchmark.py \
    --anomaly-classes hourglass --output-dir paper_scripts/v2/results/hourglass

# Quick plumbing smoke (undertrained, ~7 min)
conda run -n am python paper_scripts/v2/run_benchmark.py \
    --anomaly-classes hourglass --num-train-iter 10 \
    --label-configs '[{"n_anomaly":5,"n_nominal":495}]' \
    --output-dir /tmp/v2_smoke
```

## Output layout & re-plotting

```
<output-dir>/
  summary.csv                                  # every class × budget, all metrics
  comparative_top_n_detection.pdf              # cross-class efficiency at the final budget
  <dataset>_<class>/
    summary.csv                                # this class, all budgets
    summary_vs_label_count.png
    plots/combined_top_n_detection.pdf         # THE detection-efficiency figure (budgets overlaid)
    a<N>_n<M>/                                  # one per budget
      labeled_data.csv  model.safetensors  metrics.csv
      scores.csv.gz                            # per-image (filename, score, label_idx, true_anomaly)
      plots/*.pdf                              # score histogram, ROC, PR, top-N detection
      plots/plot_data/*.pkl                    # pickled inputs — re-plot without re-running
```

Every figure is regenerable from `scores.csv.gz` (raw scores) or the
`plot_data/*.pkl` files (exact plot inputs) — no need to re-train or re-score.

## Active-learning run (the fair v1 reproduction)

`run_active_learning.py` reproduces the paper's train → relabel → retrain loop,
which is the apples-to-apples comparison (it removes the static-label-count
approximation). Per class: start from 5 anomaly / 495 nominal, train
`--iters-per-cycle` iterations, score everything, then relabel the top-K true
anomalies (confirmed detections) and top-K true nominals (false positives) — the
same selection as v1's `find_mislabeled` — and repeat for `--cycles` cycles.
Cycle `k` lines up with v1 `iteration_k` (cycle 1 = cold start on identical
5/495 labels; final cycle = fully refined).

```bash
conda run -n am python paper_scripts/v2/run_active_learning.py \
    --anomaly-classes all --iters-per-cycle 300 --cycles 3 --n-each 10 \
    --output-dir paper_scripts/v2/results/miniimagenet_al
```

Each cycle retrains **from the pretrained backbone** (not continuing the previous
cycle's weights); pass `--iters-per-cycle 100` to match the paper's per-cycle
budget. `compare_v1_v2.py --v2-dir <the AL dir>` auto-detects the cycle structure
and overlays v2 cycle 1/last against v1 iteration 1/3.

## Comparing against the original (v1) paper results

`compare_v1_v2.py` overlays the v2 detection-efficiency curves against the
original paper run at
`/media/team_workspaces/AnomalyMatch/paper_results/FullCorrectedOutput`. Because
v2 reuses the v1 plotting code, both sides' curves are computed identically and
overlay directly. It maps **v1 iteration 1 (5-anomaly cold start) ↔ v2 budget 0**
and **v1 iteration 3 (final, after 3 AL cycles) ↔ v2 budget 2**.

```bash
conda run -n am python paper_scripts/v2/compare_v1_v2.py \
    --v2-dir paper_scripts/v2/results/miniimagenet
```

Outputs (to `/media/team_workspaces/AnomalyMatch/paper_results_v2/comparisons/`):
per-class `*_detection_efficiency_v1_vs_v2.pdf` (v1 dashed vs v2 solid),
`metrics_v1_vs_v2.csv`, and `auroc_v1_vs_v2.pdf`. Classes v2 has not finished are
skipped, so it can be run mid-sweep. v2 run outputs are archived under
`paper_results_v2/runs/`.

## Paper reference numbers (regression targets)

From the paper, after active-learning refinement:

| dataset | AUROC | AUPRC | top-1% precision |
|---------|-------|-------|------------------|
| miniImageNet | 0.96 | 0.82 | 76% |
| GalaxyMNIST  | 0.89 | 0.77 | 94% |

Note these are **post-refinement** numbers; v2 trains a single model without the
active-learning loop, so the canonical `500`-label / `200`-iter point is the
closest comparison. Treat large drops (not small differences) as regressions.

## Performance notes

- Training a 200-iter model: a few minutes (mostly model + unlabeled-pool load).
- Scoring all 65k images: ~5 min. The decode/preprocess path (fitsbolt) is
  GIL-bound at ~200 img/s, so the GPU is not the bottleneck. The prediction
  subprocess pins BLAS/OpenMP/torch to one thread each to stop the decode
  thread-pool from oversubscribing the CPU (which otherwise thrashes into a
  near-stall). A future speedup would be process-based decode parallelism.
- The paper datasets live on NFS (`/media/team_workspaces/...`); per-file reads
  are fine (~1k files/s), decode is the limiter.
