[//]: # (Copyright &#40;c&#41; European Space Agency, 2025.)
[//]: # ()
[//]: # (This file is subject to the terms and conditions defined in file 'LICENCE.txt', which)
[//]: # (is part of this source code package. No part of the package, including)
[//]: # (this file, may be copied, modified, propagated, or distributed except according to)
[//]: # (the terms contained in the file 'LICENCE.txt'.)

# Headless Mode

Run AnomalyMatch from a script, with no notebook and no UI — for batch jobs,
scheduled runs, or scoring a dataset too large to sit and watch.

Everything the UI does goes through the same two entry points you call here, so
a headless run produces the same session directory and the same score database
as an interactive one.

## Score a folder with a trained model

You need a checkpoint from a previous training run. Point the config at it and
at the data to score:

```python
import anomaly_match as am

cfg = am.get_default_cfg()
cfg.name = "batch_evaluation"

# Checkpoint from a previous training run
cfg.model_path = "/path/to/model.safetensors"

# Data to score: images, Zarr stores, or Cutana catalogues
cfg.prediction_search_dir = "/path/to/images_to_evaluate"

# `validate_config` requires both, and checks the paths exist
cfg.data_dir = "/path/to/training_images"
cfg.label_file = "/path/to/labeled_data.csv"

cfg.log_level = "INFO"

session = am.Session(cfg)
session.evaluate_all_images()
session.save_session()
```

`evaluate_all_images` blocks until every source is scored.

!!! note "Normalisation comes from the checkpoint"

    Do not re-specify `cfg.normalisation.*` for prediction. The checkpoint
    carries the fitsbolt config it was trained with, and
    `evaluate_all_images` always syncs from it — scoring on anything else
    would feed the model images it never saw. A checkpoint without an
    embedded config is rejected; retrain to embed one.

Two behaviours worth knowing before you script around this:

- **Sources in the label file are not scored.** They are already labelled, so
  the run excludes them — score 20 images with 2 of them labelled and the
  database holds 18 rows.
- **Re-running resumes.** Sources already in `predictions.db` are skipped, so
  an interrupted run continues where it stopped rather than starting over.

## Read the scores back

Results live in a SQLite database, not a CSV. Open it read-only so a query
never contends with a run still writing:

```python
from anomaly_match.prediction import AnomalyScoreDB, prediction_db_path

db_path = prediction_db_path(session.cfg)

with AnomalyScoreDB(db_path, read_only=True) as db:
    print(f"{db.get_count()} sources scored")
    for row in db.get_top_results(10):
        print(f"{row['score']:.4f}  {row['filename']}")
```

`get_results(sort_by=..., limit=..., offset=...)` pages through the full table;
`sort_by` takes `"score_desc"`, `"score_asc"`, `"score_mean_dist"`,
`"score_median_dist"`, `"updated_desc"`, `"updated_asc"` or `"random"`.

!!! warning "The live database is not always in the session directory"

    When the session directory is on network storage, the database a run
    writes to lives in local scratch instead, and is copied back into the
    session after every chunk. So `<output_dir>/predictions.db` is a snapshot
    that lags a run still in progress, and is complete only once the run has
    finished.

    Use `prediction_db_path(cfg)` for the live database and
    `session_db_path(cfg)` for the durable session copy — both from
    `anomaly_match.prediction`, and both the same file when the session
    directory is local. `is_relocated(cfg)` says which case you are in.

## Train without the UI

Training runs as a subprocess, launched through `BackendInterface` — the same
call the UI makes. It returns immediately, so wait on the process yourself:

```python
import anomaly_match as am
from anomaly_match_ui import BackendInterface

cfg = am.get_default_cfg()
cfg.name = "headless_training"
cfg.data_dir = "/path/to/training_images"
cfg.label_file = "/path/to/labeled_data.csv"
cfg.num_train_iter = 200
cfg.log_level = "INFO"

session = am.Session(cfg)
BackendInterface.set_session(session)

process, temp_dir, progress_file = BackendInterface.launch_training_subprocess()

# The subprocess logs to stderr through a pipe. Read it as it arrives — see below.
for line in process.stderr:
    print(line.decode().rstrip())
process.wait()

if process.returncode != 0:
    raise RuntimeError(f"Training failed; see {session.cfg.output_dir}/subprocess_logs/")

print(session.cfg.model_path)  # checkpoint the run produced
```

`BackendInterface` lives in the UI package but imports without a display or a
kernel — it is a thin delegation layer, not a widget.

!!! warning "You must drain `process.stderr`"

    `launch_training_subprocess` spawns with `stderr=subprocess.PIPE`, and the
    training subprocess writes its whole log there. A wait loop that never
    reads the pipe deadlocks once the run has produced about 64 KiB of
    output — the subprocess blocks writing, and `poll()` never returns. The UI
    avoids this with a reader thread; a script must read the pipe, as above, or
    use `process.communicate()`.

    Set `cfg.log_level` rather than calling `am.set_log_level` before
    `Session(cfg)`: the constructor re-applies `cfg.log_level` and drops the
    sinks an earlier call installed.

!!! warning "Check the exit code before using `cfg.model_path`"

    `launch_training_subprocess` points `cfg.model_path` at the checkpoint it
    is *about to* write, before the subprocess starts. A crashed run therefore
    leaves the same path set as a successful one, at a file that was never
    written — so a script that skips the `returncode` check happily scores
    with the previous iteration's model, or dies later on a missing file. The
    traceback is in `subprocess_logs/` and `iteration_N/training.log`, not in
    `temp_dir`, which holds only the config and labels handed to the run.

Each call writes a new `iteration_N/` directory under `cfg.output_dir` and
points `cfg.model_path` at the checkpoint inside it, so the path is ready to
hand to `evaluate_all_images` for the scoring half of the loop.

### Following progress

`progress_file` is JSON-lines, one object per event — the same file the UI
polls to draw its progress bar. Read it after the run for a summary, or poll it
during one instead of printing stderr:

```python
import json

with open(progress_file) as f:
    for line in f:
        event = json.loads(line)
        if event["status"] == "training":
            print(f"iteration {event['iteration']}/{event['total']}")
```

The final line has `status` `"done"` and carries `model_path` and `elapsed`.

## What a headless run leaves behind

The same session directory an interactive run produces — see
[Session Tracking](../getting-started.md#session-tracking) for the layout. A
scoring-only run has no `iteration_N/`, and a training-only run has no
`predictions.db`.

`session_metadata.json` only exists once `save_session()` has run, and
[`print_session`][anomaly_match.data_io.SessionIOHandler.print_session] reads
it. `save_session()` returns the session directory, so pass that straight on:

```python
session_dir = session.save_session()
am.print_session(session_dir)
```

To inspect an earlier run, pass its directory instead. Each session gets its
own folder, `anomaly_match_results/sessions/<cfg.name>_<YYYYMMDD_HHMMSS>`, so
replace the name and timestamp below with those of your run:

```python
am.print_session("anomaly_match_results/sessions/batch_evaluation_20260101_120000")
```
