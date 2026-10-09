#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Prediction pipeline profiler for structured performance metrics.

Provides context-manager-based stage timing, per-process partial JSON reports,
and a merge step that produces an aggregate ``performance_report.json``.

Uses ``time.perf_counter()`` for all timing measurements — the highest-resolution
monotonic clock available on the platform.  This is the same clock used internally
by ``timeit``.  Unlike ``timeit`` (designed for micro-benchmarking small snippets
via repeated execution), ``perf_counter`` is the standard choice for production
profiling of real workloads.
"""

import glob as glob_module
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import psutil
import torch
from loguru import logger


class _StageTimer:
    """Context manager that records elapsed time for a named profiler stage."""

    def __init__(self, profiler, name):
        self._profiler = profiler
        self._name = name
        self._start = None

    def __enter__(self):
        if self._profiler._batch_wall_start is None:
            self._profiler._batch_wall_start = time.perf_counter()
        self._start = time.perf_counter()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        elapsed = time.perf_counter() - self._start
        self._profiler._current_stages[self._name] = elapsed
        return False


class PredictionProfiler:
    """Profiles prediction pipeline stages with per-batch timing and memory tracking.

    Tracked metrics
    ---------------
    **Per-batch** (recorded via ``stage()`` context managers + ``end_batch()``):

    - ``stages``: wall-clock time (seconds) for each named stage within the batch,
      measured with ``time.perf_counter()``.
    - ``wall_clock``: total elapsed time from first stage entry to ``end_batch()``.
    - ``overhead``: ``wall_clock - sum(stages)`` — time not attributed to any stage.
    - ``batch_size``: number of images in the batch.
    - ``memory`` (every *MEMORY_SNAPSHOT_EVERY_N_BATCHES* batches):
      ``resident_set_size_mb`` (via psutil if installed), ``gpu_allocated_mb`` and
      ``gpu_reserved_mb`` (via ``torch.cuda``).

    **Per-process** (partial report via ``save_partial_report()``):

    - ``total_images``: sum of batch sizes.
    - ``total_wall_clock_s``: elapsed time from profiler creation to report save.
    - ``throughput_images_per_sec``: ``total_images / total_wall_clock_s``.
    - ``stages``: per-stage totals, percentages, and individual batch times.
    - ``peak_memory``: maximum observed values across all snapshots.
    - ``finalization_stages``: one-off stage timings (e.g. ``result_save``) recorded
      via ``record_finalization()`` — not part of batch metrics.
    - ``script``: name of the prediction script that produced this partial.
    - ``timestamp``: ISO-8601 UTC time when the report was written.

    **Aggregate** (merged report via ``Coordinator.finalize()``):

    - ``total_wall_clock_s``: sum of per-process wall clocks **plus** spawn overhead,
      giving the true end-to-end sequential time.
    - ``throughput_images_per_sec``: ``total_images / total_wall_clock_s``.
    - ``stages``: merged totals, percentages, and percentiles (p50/p95/p99)
      computed from all per-batch times across processes.
    - ``peak_memory``: element-wise max across all processes.
    - ``per_process``: per-process throughput summaries.
    - ``process_spawn``: inter-process spawn gap times and total overhead.
    - ``batch_timeseries``: sampled subset of per-batch stage times (max 200 entries,
      sampled uniformly per process x stage combination to avoid skipping groups).

    Usage::

        profiler = PredictionProfiler(output_dir="/tmp/out", script="prediction_zarr")
        for batch in batches:
            with profiler.stage("io_load"):
                data = load(batch)
            with profiler.stage("inference"):
                result = model(data)
            profiler.end_batch(batch_size=len(data))
        with profiler.stage("result_save"):
            save(result)
        profiler.record_finalization()
        profiler.save_partial_report()
    """

    PROFILING_SUBDIR = "profiling"
    MEMORY_SNAPSHOT_EVERY_N_BATCHES = 10
    PARTIAL_FILENAME_TEMPLATE = "performance_partial_{}.json"
    MERGED_FILENAME = "performance_report.json"

    def __init__(self, output_dir, process_idx=None, script=None):
        self._profiling_dir = os.path.join(output_dir, self.PROFILING_SUBDIR)
        os.makedirs(self._profiling_dir, exist_ok=True)
        if process_idx is None:
            process_idx = self._detect_next_process_idx()
        self._process_idx = process_idx
        self._script = script
        self._current_stages = {}
        self._batch_wall_start = None
        self._batch_records = []
        self._finalization_stages = {}
        self._peak_memory = {
            "resident_set_size_mb": 0.0,
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
        }
        self._start_time = time.perf_counter()
        self._total_images = 0

    def _detect_next_process_idx(self):
        """Determine the next available process index from existing partial reports."""
        pattern = os.path.join(
            self._profiling_dir,
            self.PARTIAL_FILENAME_TEMPLATE.format("*"),
        )
        existing = glob_module.glob(pattern)
        if not existing:
            return 0
        indices = []
        for path in existing:
            basename = os.path.basename(path)
            idx_str = basename.replace("performance_partial_", "").replace(".json", "")
            try:
                indices.append(int(idx_str))
            except ValueError:
                pass
        return (max(indices) + 1) if indices else 0

    def stage(self, name):
        """Return a context manager that records elapsed time for *name*."""
        return _StageTimer(self, name)

    def end_batch(self, batch_size):
        """Finalize the current batch record."""
        wall_clock = (
            time.perf_counter() - self._batch_wall_start
            if self._batch_wall_start is not None
            else 0.0
        )
        stage_sum = sum(self._current_stages.values())
        overhead = max(0.0, wall_clock - stage_sum)

        record = {
            "batch_idx": len(self._batch_records),
            "batch_size": batch_size,
            "wall_clock": wall_clock,
            "overhead": overhead,
            "stages": dict(self._current_stages),
        }

        # Memory snapshot every N batches
        if len(self._batch_records) % self.MEMORY_SNAPSHOT_EVERY_N_BATCHES == 0:
            mem = self._take_memory_snapshot()
            record["memory"] = mem
            for key in self._peak_memory:
                if mem.get(key) is not None:
                    self._peak_memory[key] = max(self._peak_memory[key], mem[key])

        self._batch_records.append(record)
        self._total_images += batch_size
        self._current_stages = {}
        self._batch_wall_start = None

    def record_finalization(self):
        """Record current stages as one-off finalization metrics.

        Call this instead of ``end_batch()`` for post-loop operations like
        ``result_save`` that should not inflate batch counts or throughput.
        """
        self._finalization_stages.update(self._current_stages)
        self._current_stages = {}
        self._batch_wall_start = None

    def save_partial_report(self):
        """Write a per-process partial JSON report to the profiling subdirectory."""
        total_wall = time.perf_counter() - self._start_time
        throughput = self._total_images / total_wall if total_wall > 0 else 0.0

        # Aggregate per-stage totals
        stage_totals = {}
        stage_batch_times = {}
        for rec in self._batch_records:
            for stage_name, elapsed in rec["stages"].items():
                stage_totals[stage_name] = stage_totals.get(stage_name, 0.0) + elapsed
                stage_batch_times.setdefault(stage_name, []).append(elapsed)

        total_stage_time = sum(stage_totals.values())
        stages = {}
        for name, total_s in stage_totals.items():
            pct = (total_s / total_stage_time * 100) if total_stage_time > 0 else 0.0
            stages[name] = {
                "total_s": round(total_s, 6),
                "percentage": round(pct, 2),
                "batch_times": [round(t, 6) for t in stage_batch_times[name]],
            }

        report = {
            "process_idx": self._process_idx,
            "total_images": self._total_images,
            "total_wall_clock_s": round(total_wall, 6),
            "throughput_images_per_sec": round(throughput, 2),
            "stages": stages,
            "peak_memory": {k: round(v, 2) for k, v in self._peak_memory.items()},
            "num_batches": len(self._batch_records),
            "batch_sizes": [r["batch_size"] for r in self._batch_records],
            "script": self._script,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self._finalization_stages:
            report["finalization_stages"] = {
                k: round(v, 6) for k, v in self._finalization_stages.items()
            }

        path = os.path.join(
            self._profiling_dir,
            self.PARTIAL_FILENAME_TEMPLATE.format(self._process_idx),
        )
        # The profiling directory is created in __init__, but a concurrent
        # process may rmtree the session dir (label cache rebuilds, session
        # rotation, user clearing output_dir) between profiler construction
        # and report save at end-of-batch.  Re-create defensively so the
        # subprocess doesn't exit non-zero on an otherwise-successful run.
        os.makedirs(self._profiling_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Profiler partial report saved to {path}")
        return path

    @staticmethod
    def merge_reports(profiling_dir, spawn_times=None):
        """Merge per-process partial reports into a single aggregate report."""
        pattern = os.path.join(
            profiling_dir,
            PredictionProfiler.PARTIAL_FILENAME_TEMPLATE.format("*"),
        )
        partial_files = sorted(glob_module.glob(pattern))

        if not partial_files:
            report = {
                "total_images": 0,
                "total_wall_clock_s": 0.0,
                "throughput_images_per_sec": 0.0,
                "stages": {},
                "per_process": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            os.makedirs(profiling_dir, exist_ok=True)
            out_path = os.path.join(profiling_dir, PredictionProfiler.MERGED_FILENAME)
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            return report

        partials = []
        for pf in partial_files:
            with open(pf) as f:
                partials.append(json.load(f))
        partials.sort(key=lambda p: p["process_idx"])

        # Aggregate totals — include spawn overhead for true end-to-end wall time
        total_images = sum(p["total_images"] for p in partials)
        total_process_wall = sum(p["total_wall_clock_s"] for p in partials)
        spawn_overhead = sum(spawn_times) if spawn_times else 0.0
        total_wall = total_process_wall + spawn_overhead
        throughput = total_images / total_wall if total_wall > 0 else 0.0

        # Aggregate stages
        all_stage_names = set()
        for p in partials:
            all_stage_names.update(p["stages"].keys())

        overall_total = sum(sum(s["total_s"] for s in p["stages"].values()) for p in partials)

        stages = {}
        for name in sorted(all_stage_names):
            combined_times = []
            total_s = 0.0
            for p in partials:
                if name in p["stages"]:
                    total_s += p["stages"][name]["total_s"]
                    combined_times.extend(p["stages"][name]["batch_times"])

            percentiles = {}
            if combined_times:
                arr = np.array(combined_times)
                for pct in (50, 95, 99):
                    percentiles[f"p{pct}"] = round(float(np.percentile(arr, pct)), 6)

            pct = (total_s / overall_total * 100) if overall_total > 0 else 0.0

            stages[name] = {
                "total_s": round(total_s, 6),
                "percentage": round(pct, 2),
                "percentiles": percentiles,
                "num_batches": len(combined_times),
            }

        # Aggregate finalization stages
        finalization_stages = {}
        for p in partials:
            for name, elapsed in p.get("finalization_stages", {}).items():
                finalization_stages[name] = finalization_stages.get(name, 0.0) + elapsed

        # Aggregate peak memory
        peak_memory = {
            "resident_set_size_mb": 0.0,
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
        }
        for p in partials:
            for key in peak_memory:
                peak_memory[key] = max(peak_memory[key], p.get("peak_memory", {}).get(key, 0.0))

        # Spawn times
        process_spawn = None
        if spawn_times:
            process_spawn = {
                "spawn_times_s": [round(t, 6) for t in spawn_times],
                "total_spawn_overhead_s": round(spawn_overhead, 6),
            }

        # Per-process summaries
        per_process = []
        for p in partials:
            per_process.append(
                {
                    "process_idx": p["process_idx"],
                    "total_images": p["total_images"],
                    "total_wall_clock_s": p["total_wall_clock_s"],
                    "throughput_images_per_sec": p["throughput_images_per_sec"],
                }
            )

        # Batch timeseries (sampled uniformly per process x stage combination)
        all_batch_times = _build_batch_timeseries(partials, max_entries=200)

        report = {
            "total_images": total_images,
            "total_wall_clock_s": round(total_wall, 6),
            "throughput_images_per_sec": round(throughput, 2),
            "stages": stages,
            "peak_memory": {k: round(v, 2) for k, v in peak_memory.items()},
            "per_process": per_process,
            "batch_timeseries": all_batch_times,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if finalization_stages:
            report["finalization_stages"] = {k: round(v, 6) for k, v in finalization_stages.items()}
        if process_spawn is not None:
            report["process_spawn"] = process_spawn

        out_path = os.path.join(profiling_dir, PredictionProfiler.MERGED_FILENAME)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Profiler merged report saved to {out_path}")
        return report

    @staticmethod
    def cleanup_partial_reports(profiling_dir):
        """Remove partial report files, preserving the merged report."""
        pattern = os.path.join(
            profiling_dir,
            PredictionProfiler.PARTIAL_FILENAME_TEMPLATE.format("*"),
        )
        for path in glob_module.glob(pattern):
            os.remove(path)

    @staticmethod
    def _take_memory_snapshot():
        """Capture current RSS and GPU memory usage."""
        snapshot = {
            "resident_set_size_mb": None,
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
        }
        try:
            process = psutil.Process()
            snapshot["resident_set_size_mb"] = process.memory_info().rss / 1024**2
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                snapshot["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                snapshot["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        except Exception:
            pass
        return snapshot

    class Coordinator:
        """Coordinates profiler reports across sequential subprocess runs.

        Tracks inter-process spawn timing and provides a single ``finalize()``
        call that merges partial reports, cleans up intermediates, and logs
        the aggregate report path.

        Usage in ``session.py``::

            coordinator = PredictionProfiler.Coordinator(output_dir)
            for file_idx, input_file in enumerate(input_files):
                coordinator.record_subprocess_gap()
                run_pipeline(...)
                coordinator.mark_subprocess_complete()
            coordinator.finalize()
        """

        def __init__(self, output_dir):
            self._output_dir = output_dir
            self._profiling_dir = os.path.join(output_dir, PredictionProfiler.PROFILING_SUBDIR)
            self._spawn_times = []
            self._last_subprocess_end = None

        def record_subprocess_gap(self):
            """Record the time gap since the last subprocess completed."""
            if self._last_subprocess_end is not None:
                self._spawn_times.append(time.perf_counter() - self._last_subprocess_end)

        def mark_subprocess_complete(self):
            """Mark the current subprocess as completed."""
            self._last_subprocess_end = time.perf_counter()

        def finalize(self):
            """Merge partial reports, clean up intermediates, and return the report."""
            report = PredictionProfiler.merge_reports(
                self._profiling_dir, spawn_times=self._spawn_times
            )
            PredictionProfiler.cleanup_partial_reports(self._profiling_dir)
            return report


def _build_batch_timeseries(partials, max_entries=200):
    """Build a sampled batch timeseries, sampling uniformly per (process, stage).

    Stride-based sampling on the flat list can skip entire stages or processes.
    Instead, allocate budget proportionally across groups and sample within each.
    """
    groups = {}
    for p in partials:
        for stage_name, stage_data in p["stages"].items():
            key = (p["process_idx"], stage_name)
            entries = [
                {
                    "process_idx": p["process_idx"],
                    "batch_idx": i,
                    "stage": stage_name,
                    "time_s": t,
                }
                for i, t in enumerate(stage_data["batch_times"])
            ]
            groups[key] = entries

    if not groups:
        return []

    total_entries = sum(len(v) for v in groups.values())
    if total_entries <= max_entries:
        result = []
        for entries in groups.values():
            result.extend(entries)
        return result

    # Sample proportionally per group, with at least 1 per group
    per_group_budget = max(1, max_entries // len(groups))
    result = []
    for entries in groups.values():
        if len(entries) <= per_group_budget:
            result.extend(entries)
        else:
            step = len(entries) / per_group_budget
            indices = [int(i * step) for i in range(per_group_budget)]
            result.extend(entries[idx] for idx in indices)

    return result[:max_entries]
