[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Training

## Overview

AnomalyMatch uses the FixMatch semi-supervised learning algorithm with an
active learning loop. The training workflow is a repeating cycle:

```
Setup -> Train -> Score -> Browse & Label -> Retrain -> ...
```

Both training and scoring run as **subprocesses**, keeping the UI responsive
and memory-isolated between cycles.

## Training Setup Screen

The workflow starts with a setup screen where you select:

1. **Source folder** containing your images (JPEG, PNG, FITS, Zarr, or Cutana catalogues)
2. **Label file** (`labeled_data.csv`) mapping filenames to `normal` / `anomaly` labels
3. **Metadata file** (optional) with RA/Dec coordinates for ESASky integration
4. **Model checkpoint** (optional, `.safetensors`) to resume from a previous training run
5. **Training iterations** (50-600, default 200)
6. **Stratify source size per tile** (Cutana sources only) — see below

Click **Start Training** to begin.

### Source-size stratification (Cutana only)

Euclid Q1/DR1 catalogues are dominated by small sources (diameter median ~12px
with a long tail), so an unstratified unlabelled pool is almost entirely tiny
sources. Enabling **Stratify source size per tile** instead draws the unlabelled
pool **uniformly in log source size** (up to `cutana_size_stratify_max_px`,
default 200px): the diameters are binned in log space
(`cutana_size_stratify_bins`) and each bin contributes the *same* number of
sources wherever the catalogue can supply it — a flat, uniform-in-size pool
rather than one that mirrors the real (tiny-source-dominated) distribution. The
aim is even representation across sizes, **not** to reproduce the underlying
distribution. Large sources above `max_px` are all kept (never cut off), so the
pool thins there only because such sources are genuinely rare — every one
available is taken. The toggle only applies to Cutana catalogues (which carry a
per-source `diameter_pixel`/`diameter_arcsec`); it is disabled for image folders
and Zarr sources.

The checkbox is also present on the training screen, but its value is read
**only when a training subprocess launches** — i.e. when you click **Start
Training** or **Retrain**. Toggling it while browsing a scored gallery has no
effect until the next retrain; it does not re-sample the current pool live.

## Training Cycle

### 1. Training

A background subprocess trains the FixMatch model:

- Loads labeled images from the label CSV
- Loads N unlabeled images from the source folder (auto-computed to fill
  training iterations, capped at 10K-20K depending on image resolution)
- Trains for the configured number of iterations
- Saves the model checkpoint to the session directory
- Progress is shown via iteration count and progress bar

### 2. Scoring

After training completes, the entire source folder is scored automatically:

- Reuses the existing prediction infrastructure (`prediction_process.py`)
- Results are written to an `AnomalyScoreDB` (SQLite)
- Gallery populates as results arrive — you can start browsing immediately
- Progress bar shows images scored, speed, and ETA

### 3. Browse and Label

The gallery displays scored images with interactive labeling:

- **Paginated grid** of thumbnail images sorted by anomaly score
- **Sort modes**: Highest Score, Lowest Score, Closest to Mean, Closest to Median, Most Recent, Random
- **Three-way label toggle** on each thumbnail: Anomaly (red) / Unlabelled (grey) / Normal (green)
- **Score histogram** on the right panel showing the overall score distribution
- **Label summary** showing how many new labels you've added
- Click any thumbnail to open the full-size image detail view with transform controls

Labels persist across page navigation and are stored in a session-level dictionary.

### 4. Retrain

Click **Retrain** to:

1. Merge your new gallery labels with the existing `labeled_data.csv`
2. Save the updated label file
3. Loop back to the Training state with the expanded label set

This cycle repeats — each iteration uses more labeled data, improving the model.

### Choosing the number of iterations

Each training run loads a fresh unlabelled pool. Its size is

```
N_unlabeled = min(batch_size × uratio × num_train_iter, cap)
```

where `cap` is `cfg.unlabeled_pool_cap` (default **20 000**, images ≤ 200px) or
`cfg.unlabeled_pool_cap_hires` (default **10 000**, images > 200px, selected by
`cfg.unlabeled_pool_hires_threshold`). This `min` is applied in
`subprocess_scripts/training_process.py` (`n_unlabeled = min(…, cap)`).

**The cap binds at a fairly low iteration count.** With the defaults
(`batch_size = 16`, `uratio = 5`) the pool grows by 80 sources per iteration, so
it reaches the 20 000 cap at **250 iterations** — and the low-res cap is *below*
the top of the 50–600 iterations slider. Past that point extra iterations do
**not** enlarge the pool: they add more passes over the *same* capped set of
sources, not new ones. (Raising `uratio` reaches the cap sooner — e.g.
`uratio = 10` hits it at 125 iterations.)

The pool is sampled deterministically from `cfg.seed` mixed with a hash of the
current **labelled** set, so:

- **Retraining after adding labels** (the normal active-learning loop) draws a
  **fresh** unlabelled pool each cycle — the label set changed, so the hash and
  therefore the sample change.
- **Retraining without adding new labels** reproduces the **same** pool — the
  sampling is stable for a given label set (reproducibility by design).

Practical guidance — and *why*: prefer several shorter cycles with labelling in
between over one long run of the same total length. The reason is the cap
combined with per-cycle resampling. Because each cycle re-draws an independent,
up-to-`cap` pool (its label hash changed), *N* cycles expose the model to up to
*N × cap* distinct unlabelled sources, whereas a single run is clamped to one
`cap`-sized pool no matter how long you train it. So above the cap, `5 × 100`
iterations sees markedly more of the unlabelled data than `1 × 500` (which,
past 250 iters, is just re-reading the same 20 000 sources). Below the cap the
two cover a comparable number of distinct sources — the shorter-cycles advantage
is specifically a consequence of the cap plus resampling, not of iteration count
alone — but shorter cycles still win on labelling cadence. The trade-off is
re-running dataset load and scoring each cycle. The default of 200 iterations is
a reasonable balance; raise it (up to the cap) when the pool is large and
diverse, lower it to label more frequently.

## Training from a script

The Training screen drives a subprocess through `BackendInterface`, and
`BackendInterface.launch_training_subprocess()` is callable directly to train
without a notebook. It returns as soon as the subprocess starts, handing back
the process, its temporary directory and a JSON-lines progress file.

See [Headless Mode](headless.md) for the full script — registering the session,
waiting for the run, and checking the exit code before using the checkpoint it
produced.

## Controls

| Control | Description |
|---------|-------------|
| **Iterations slider** | Number of FixMatch training iterations (50-600) |
| **Retrain** | Merge labels and start a new training cycle |
| **Stop** | Kill the current training/scoring subprocess |
| **Save Model** | Save the current model checkpoint |
| **Save Labels** | Save the current label set to CSV |

## Key Configuration

Training parameters can be set in the config before session creation, or
left as defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_iter` | 200 | FixMatch iterations per training cycle |
| `batch_size` | 16 | Training batch size |
| `lr` | 0.0075 | Learning rate |
| `ema_m` | 0.99 | EMA momentum for the evaluation model |
| `p_cutoff` | 0.95 | Pseudo-label confidence threshold |
| `ulb_loss_ratio` | 1.0 | Weight for unlabeled loss |
| `net` | `efficientnet-lite0` | Backbone architecture |
| `cutana_stratify_source_size` | `False` | Sample the Cutana unlabelled pool ~uniform in (log) source size |
| `cutana_max_unlabeled_tiles` | 16 | FITS tile sets to draw the unlabelled pool from (more tiles supply the rare large sources) |
| `cutana_size_stratify_bins` | 30 | Log-spaced size bins spanning the flattened region `[min, max_px]` |
| `cutana_size_stratify_max_px` | 200 | Top of the flattened (uniform) log-size range, in px; larger sources are all kept but grow scarce, so the pool thins there by availability, not by design |

See the [Configuration Reference](configuration.md) for the full parameter list.

## Subprocess Architecture

Training and scoring run as isolated child processes for:

- **Memory isolation** — GPU memory is freed between cycles, preventing leaks
- **UI responsiveness** — the notebook never blocks
- **Clean interrupts** — Stop button kills the subprocess immediately
- **Logging** — each subprocess writes a timestamped log to `subprocess_logs/` in the session directory, and training also writes `iteration_N/training.log`

The UI monitors progress via a JSON-lines file (training) or polling the
SQLite database (scoring).
