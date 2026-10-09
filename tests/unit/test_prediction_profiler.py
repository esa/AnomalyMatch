#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import json
import os
import time

import pytest

from anomaly_match.utils.prediction_profiler import PredictionProfiler


class TestStageTimer:
    def test_stage_records_elapsed_time(self, tmp_path):
        profiler = PredictionProfiler(output_dir=str(tmp_path), process_idx=0)
        with profiler.stage("test_stage"):
            time.sleep(0.01)
        assert "test_stage" in profiler._current_stages
        assert profiler._current_stages["test_stage"] >= 0.01

    def test_stage_handles_exception(self, tmp_path):
        profiler = PredictionProfiler(output_dir=str(tmp_path), process_idx=0)
        with pytest.raises(ValueError):
            with profiler.stage("failing"):
                raise ValueError("boom")
        assert "failing" in profiler._current_stages
        assert profiler._current_stages["failing"] >= 0.0


class TestEndBatch:
    def test_end_batch_creates_record_and_tracks_total(self, tmp_path):
        profiler = PredictionProfiler(output_dir=str(tmp_path), process_idx=0)
        for _ in range(3):
            with profiler.stage("io"):
                pass
            profiler.end_batch(batch_size=32)
        assert len(profiler._batch_records) == 3
        assert profiler._batch_records[0]["batch_size"] == 32
        assert profiler._total_images == 96
        # Current stages should be cleared after end_batch
        assert profiler._current_stages == {}


class TestSavePartialReport:
    def test_partial_report_structure(self, tmp_path):
        profiler = PredictionProfiler(
            output_dir=str(tmp_path), process_idx=1, script="prediction_zarr"
        )
        for _ in range(2):
            with profiler.stage("io"):
                time.sleep(0.005)
            with profiler.stage("inference"):
                time.sleep(0.005)
            profiler.end_batch(batch_size=10)

        with profiler.stage("result_save"):
            time.sleep(0.005)
        profiler.record_finalization()

        path = profiler.save_partial_report()
        assert os.path.exists(path)

        with open(path) as f:
            report = json.load(f)

        assert report["process_idx"] == 1
        assert report["total_images"] == 20
        assert report["script"] == "prediction_zarr"
        assert "total_wall_clock_s" in report
        assert "throughput_images_per_sec" in report
        assert report["stages"]["io"]["total_s"] >= 0.01
        assert len(report["stages"]["io"]["batch_times"]) == 2
        assert "finalization_stages" in report
        assert "result_save" in report["finalization_stages"]

    def test_zero_batches_report(self, tmp_path):
        profiler = PredictionProfiler(output_dir=str(tmp_path), process_idx=0)
        path = profiler.save_partial_report()
        with open(path) as f:
            report = json.load(f)
        assert report["total_images"] == 0
        assert report["num_batches"] == 0
        assert report["stages"] == {}


class TestProcessIdxAutoDetection:
    def test_auto_detection_increments(self, tmp_path):
        profiler0 = PredictionProfiler(output_dir=str(tmp_path))
        assert profiler0._process_idx == 0

        with profiler0.stage("io"):
            pass
        profiler0.end_batch(batch_size=5)
        profiler0.save_partial_report()

        profiler1 = PredictionProfiler(output_dir=str(tmp_path))
        assert profiler1._process_idx == 1

    def test_explicit_idx_overrides_detection(self, tmp_path):
        profiler = PredictionProfiler(output_dir=str(tmp_path), process_idx=42)
        assert profiler._process_idx == 42


class TestMergeReports:
    def _create_partial(self, tmp_path, process_idx, total_images, stages):
        """Helper to create a partial report file in the profiling subdir."""
        profiling_dir = os.path.join(str(tmp_path), "profiling")
        os.makedirs(profiling_dir, exist_ok=True)
        report = {
            "process_idx": process_idx,
            "total_images": total_images,
            "total_wall_clock_s": 10.0,
            "throughput_images_per_sec": total_images / 10.0,
            "stages": stages,
            "peak_memory": {
                "resident_set_size_mb": 100.0,
                "gpu_allocated_mb": 50.0,
                "gpu_reserved_mb": 80.0,
            },
            "num_batches": 2,
            "batch_sizes": [total_images // 2, total_images // 2],
            "script": "test",
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        path = os.path.join(
            profiling_dir,
            PredictionProfiler.PARTIAL_FILENAME_TEMPLATE.format(process_idx),
        )
        with open(path, "w") as f:
            json.dump(report, f)
        return report

    def test_merge_two_processes(self, tmp_path):
        stages = {
            "io": {"total_s": 3.0, "percentage": 60.0, "batch_times": [1.5, 1.5]},
            "inference": {
                "total_s": 2.0,
                "percentage": 40.0,
                "batch_times": [1.0, 1.0],
            },
        }
        self._create_partial(tmp_path, 0, 100, stages)
        self._create_partial(tmp_path, 1, 200, stages)

        profiling_dir = os.path.join(str(tmp_path), "profiling")
        merged = PredictionProfiler.merge_reports(profiling_dir)
        assert merged["total_images"] == 300
        assert len(merged["per_process"]) == 2

    def test_merge_throughput_includes_spawn_overhead(self, tmp_path):
        stages = {"io": {"total_s": 1.0, "percentage": 100.0, "batch_times": [1.0]}}
        self._create_partial(tmp_path, 0, 100, stages)

        profiling_dir = os.path.join(str(tmp_path), "profiling")
        merged = PredictionProfiler.merge_reports(profiling_dir, spawn_times=[5.0])
        # total_wall = 10.0 (process) + 5.0 (spawn) = 15.0
        assert merged["total_wall_clock_s"] == pytest.approx(15.0, abs=0.01)
        assert merged["throughput_images_per_sec"] == pytest.approx(6.67, abs=0.1)

    def test_merge_empty_directory(self, tmp_path):
        profiling_dir = os.path.join(str(tmp_path), "profiling")
        os.makedirs(profiling_dir, exist_ok=True)
        merged = PredictionProfiler.merge_reports(profiling_dir)
        assert merged["total_images"] == 0
        assert merged["per_process"] == []


class TestCleanupPartialReports:
    def test_cleanup_removes_partials_preserves_merged(self, tmp_path):
        profiling_dir = os.path.join(str(tmp_path), "profiling")
        os.makedirs(profiling_dir, exist_ok=True)
        for i in range(3):
            path = os.path.join(
                profiling_dir,
                PredictionProfiler.PARTIAL_FILENAME_TEMPLATE.format(i),
            )
            with open(path, "w") as f:
                json.dump({}, f)

        merged_path = os.path.join(profiling_dir, PredictionProfiler.MERGED_FILENAME)
        with open(merged_path, "w") as f:
            json.dump({"total_images": 100}, f)

        PredictionProfiler.cleanup_partial_reports(profiling_dir)

        import glob as glob_module

        remaining = glob_module.glob(os.path.join(profiling_dir, "performance_partial_*.json"))
        assert len(remaining) == 0
        assert os.path.exists(merged_path)
