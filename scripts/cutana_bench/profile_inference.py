#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Faithful end-to-end profiler for the Cutana prediction inference path.

Runs the *production* ``evaluate_images_from_cutana`` (so any change to
``load_model`` / ``process_batch_predictions`` / the prefetch loop is exercised)
on a real q1 catalogue and reports the per-stage breakdown the PredictionProfiler
emits, plus a score summary so accuracy-affecting changes (AMP) are visible.

Usage:
    python scripts/cutana_bench/profile_inference.py --catalogue <parquet> --tag before
"""

import argparse
import glob
import json
import os
import sqlite3
import sys
import time

import numpy as np
from dotmap import DotMap
from loguru import logger

# Resolve the live Cutana working copy for both this process and the worker
# subprocesses it spawns (they inherit PYTHONPATH). Must precede `import cutana`.
CUTANA_REPO = "/media/team_workspaces/AnomalyMatch-IDR1-Search/Cutana"
sys.path.insert(0, CUTANA_REPO)
os.environ["PYTHONPATH"] = CUTANA_REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "subprocess_scripts"))
sys.path.insert(0, _REPO)

from prediction_process_cutana import evaluate_images_from_cutana  # noqa: E402

from anomaly_match.data_io.checkpoint_io import load_checkpoint  # noqa: E402
from anomaly_match.prediction.anomaly_score_db import SCHEMA_VERSION  # noqa: E402
from prediction_utils import estimate_batch_size  # noqa: E402

MODEL_PATH = (
    "/media/home/my_workspace/AnomalyMatch/anomaly_match_results/sessions/"
    "MyRun_20260429_163254/iteration_4/model.safetensors"
)

# (n_out, n_in) channel mix for visnir3: VIS passthrough; NIR H/Y/J blended into
# the two non-VIS output channels with NIR-Y shared — matches bench_cutana.py.
VISNIR3_COMBINATION = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0],
    ],
    dtype=np.float32,
)


def build_cfg(output_dir: str, model_path: str = MODEL_PATH) -> DotMap:
    """Build the visnir3 (3-channel) prediction config used for profiling."""
    import anomaly_match as am  # noqa: PLC0415 - heavy import, only needed here

    cfg = am.get_default_cfg()
    cfg.model_path = model_path
    cfg.gpu = 0
    cfg.net = "efficientnet-lite0"
    cfg.pretrained = False
    cfg.output_dir = output_dir
    cfg.normalisation.image_size = [180, 180]
    cfg.normalisation.normalisation_method = am.NormalisationMethod.ASINH
    cfg.normalisation.output_dtype = np.uint8
    cfg.normalisation.cutout_padding_factor = 1.0
    cfg.normalisation.fits_extension = [0, 1, 2, 3]
    cfg.normalisation.n_output_channels = 3
    cfg.normalisation.channel_combination = VISNIR3_COMBINATION
    cfg.num_channels = 3

    ckpt = load_checkpoint(model_path, device="cpu")
    fb = DotMap(ckpt["fitsbolt_cfg"].toDict(), _dynamic=False)
    fb.size = [180, 180]
    cfg.fitsbolt_cfg = fb
    return cfg


def summarise(output_dir: str, tag: str, wall: float) -> None:
    """Print the profiler stage breakdown and a score summary for *tag*."""
    parts = sorted(
        glob.glob(os.path.join(output_dir, "**", "performance_partial_*.json"), recursive=True)
    )
    logger.info("===== RESULT [{}] (driver wall {:.1f}s) =====", tag, wall)
    for p in parts:
        with open(p) as f:
            rep = json.load(f)
        logger.info(
            "  {} imgs | {:.1f} img/s | wall {:.1f}s",
            rep["total_images"],
            rep["throughput_images_per_sec"],
            rep["total_wall_clock_s"],
        )
        for name, st in rep["stages"].items():
            logger.info("    {:<11} {:>7.1f}s  {:>5.1f}%", name, st["total_s"], st["percentage"])

    db_path = os.path.join(output_dir, "predictions.db")
    if os.path.isfile(db_path):
        con = sqlite3.connect(db_path)
        try:
            # This reads the score column raw, so it bypasses the AnomalyScoreDB
            # schema gate.  A pre-v4 DB holds fp16 patterns in the same INTEGER
            # column and would decode to plausible-looking garbage, so check the
            # version here rather than print numbers nobody can trust.
            row = con.execute(
                "SELECT value FROM run_metadata WHERE key = 'schema_version'"
            ).fetchone()
            stored = json.loads(row[0]) if row else None
            if stored != SCHEMA_VERSION:
                logger.warning(
                    "  skipping score summary: {} has schema_version={}, expected {}",
                    db_path,
                    stored,
                    SCHEMA_VERSION,
                )
                return
            tbl = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            # The scores live in the only data table; read it generically.
            for (name,) in tbl:
                cols = [c[1] for c in con.execute(f"PRAGMA table_info({name})").fetchall()]
                score_col = next((c for c in cols if "score" in c.lower()), None)
                if score_col is None:
                    continue
                rows = con.execute(f"SELECT {score_col} FROM {name}").fetchall()
                # Scores are the fp32 bit pattern stored as a uint32 integer.
                raw = np.array([r[0] for r in rows], dtype="<u4")
                vals = raw.view("<f4").astype(np.float64)
                if vals.size:
                    logger.info(
                        "  scores[{}]: n={} mean={:.5f} std={:.5f} frac>0.5={:.4f}",
                        name,
                        vals.size,
                        vals.mean(),
                        vals.std(),
                        float((vals > 0.5).mean()),
                    )
        finally:
            con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="/dev/shm/am_profile/subset_100k.parquet")
    ap.add_argument("--model", default=MODEL_PATH, help="model checkpoint to profile")
    ap.add_argument("--out", default=None, help="output dir (default: fresh per-tag temp dir)")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    out = args.out or f"/dev/shm/am_profile/out_{args.tag}"
    # Fresh output each run: a reused predictions.db would mark sources already
    # scored and the second run would measure nothing.
    if os.path.isdir(out):
        import shutil  # noqa: PLC0415

        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    cfg = build_cfg(out, model_path=args.model)
    batch_size = estimate_batch_size(cfg)
    logger.info("Profiling [{}]: catalogue={} batch_size={}", args.tag, args.catalogue, batch_size)

    t0 = time.time()
    evaluate_images_from_cutana(args.catalogue, cfg, batch_size=batch_size, top_n=1000)
    wall = time.time() - t0
    summarise(out, args.tag, wall)


if __name__ == "__main__":
    main()
