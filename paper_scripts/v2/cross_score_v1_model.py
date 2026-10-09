#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Decisive regression test: score v1's trained model through the v2 pipeline.

Builds a *hybrid* checkpoint = a v2 checkpoint's safetensors metadata (fitsbolt
normalisation config, net, channel count — the v2 preprocessing settings) with
v1's trained ``eval_model`` weights swapped in. Scoring that hybrid through the
identical v2 prediction path isolates where the regression lives:

- If v1's weights scored via v2 ≈ v1's reported AUROC (0.96) → **v2 training**
  is the regression (scoring/preprocessing is faithful).
- If v1's weights scored via v2 ≈ 0.81 → **v2 scoring/preprocessing** is the
  regression (the model is fine, the pipeline mis-scores it).

Run under the ``am`` env.
"""

import json
import os
import sys
import types

import numpy as np
import pandas as pd
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# v1 pickled the NormalisationMethod enum under a module path that no longer
# exists (it now lives in fitsbolt). Register a compat alias before torch.load.
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod  # noqa: E402

_shim = types.ModuleType("anomaly_match.image_processing.NormalisationMethod")
_shim.NormalisationMethod = NormalisationMethod
sys.modules["anomaly_match.image_processing.NormalisationMethod"] = _shim

import benchmark_datasets as bd  # noqa: E402
import torch  # noqa: E402
from benchmark_config import BenchmarkParams, build_prediction_cfg  # noqa: E402
from benchmark_metrics import evaluate_performance  # noqa: E402
from run_benchmark import list_image_files, predict_scores  # noqa: E402

V1_DIR = (
    "/media/team_workspaces/AnomalyMatch/paper_results/FullCorrectedOutput/"
    "miniimagenet/mini_class57_ratio0.01/miniimagenet_anomaly57_n495_a5"
)
V1_MODELS = {
    # cycle -> (v1 .pth, v1 reported AUROC for the matching iteration)
    1: (f"{V1_DIR}/benchmark_miniimagenet_anomaly57_20250726_173340/model_iteration_0.pth", 0.9614),
    3: (f"{V1_DIR}/benchmark_miniimagenet_anomaly57_20250726_173340/model_iteration_2.pth", 0.9702),
}
# A finished v2 checkpoint that carries valid fitsbolt metadata to graft onto.
V2_TEMPLATE = os.path.join(
    _THIS_DIR, "results/miniimagenet_al/miniimagenet_hourglass/cycle_1/model.safetensors"
)
OUT_DIR = os.path.join(_THIS_DIR, "results/cross_score")


def build_hybrid_checkpoint(v1_pth, v2_template, out_path):
    """Write a safetensors checkpoint = v2 metadata + v1 eval_model weights.

    Args:
        v1_pth: Path to a v1 ``.pth`` checkpoint (``eval_model`` state dict).
        v2_template: Path to a v2 ``.safetensors`` checkpoint whose metadata and
            ``train_model`` tensors are kept (only ``eval_model.*`` is replaced).
        out_path: Destination ``.safetensors`` path.

    Raises:
        KeyError: If the v1 and v2 ``eval_model`` parameter names do not match.
    """
    v1_state = torch.load(v1_pth, weights_only=False, map_location="cpu")
    v1_eval = v1_state["eval_model"]

    with safe_open(v2_template, framework="pt") as f:
        metadata = f.metadata() or {}
    v2_tensors = st_load(v2_template)

    v2_eval_names = {k[len("eval_model.") :] for k in v2_tensors if k.startswith("eval_model.")}
    v1_eval_names = set(v1_eval.keys())
    if v2_eval_names != v1_eval_names:
        only_v1 = sorted(v1_eval_names - v2_eval_names)[:5]
        only_v2 = sorted(v2_eval_names - v1_eval_names)[:5]
        raise KeyError(f"eval_model key mismatch. v1-only e.g. {only_v1}; v2-only e.g. {only_v2}")

    out = dict(v2_tensors)
    for name, tensor in v1_eval.items():
        out[f"eval_model.{name}"] = tensor.contiguous()
    # Also mirror into train_model so the file is self-consistent (unused for scoring).
    for name, tensor in v1_eval.items():
        out[f"train_model.{name}"] = tensor.contiguous()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    st_save(out, out_path, metadata=metadata)
    logger.info(f"Hybrid checkpoint written: {out_path}")
    logger.info(f"  net={json.loads(metadata.get('net', 'null'))}")
    logger.info(f"  fitsbolt_cfg present: {metadata.get('fitsbolt_cfg', 'null') != 'null'}")


def main():
    """Build hybrid checkpoints for v1 cycles 1 & 3, score via v2, report AUROC."""
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset = bd.REGISTRY["miniimagenet"]
    anomaly_idx = bd.resolve_anomaly_class(dataset, "hourglass")
    gt_df = bd.load_ground_truth(dataset)
    image_files = list_image_files(dataset.image_dir)
    params = BenchmarkParams()

    rows = []
    for cycle, (v1_pth, v1_auroc) in V1_MODELS.items():
        logger.info(f"===== cross-score v1 cycle {cycle} (reported AUROC {v1_auroc}) =====")
        hybrid = os.path.join(OUT_DIR, f"hybrid_v1_iter{cycle}.safetensors")
        build_hybrid_checkpoint(v1_pth, V2_TEMPLATE, hybrid)

        pred_dir = os.path.join(OUT_DIR, f"pred_iter{cycle}")
        os.makedirs(pred_dir, exist_ok=True)
        pred_cfg = build_prediction_cfg(hybrid, pred_dir, dataset.image_dir, params)
        pairs = predict_scores(pred_cfg, image_files, pred_dir, params.decode_workers)

        filenames = [fn for fn, _ in pairs]
        scores = np.array([s for _, s in pairs])
        metrics = evaluate_performance(scores, filenames, gt_df, anomaly_idx)
        logger.info(
            f"v1 cycle {cycle} scored via v2: AUROC={metrics['auroc']:.4f} "
            f"AUPRC={metrics['auprc']:.4f} (v1 reported {v1_auroc})"
        )
        rows.append(
            {
                "cycle": cycle,
                "v1_reported_auroc": v1_auroc,
                "v2pipeline_auroc": metrics["auroc"],
                "v2pipeline_auprc": metrics["auprc"],
                "top_1pct_precision": metrics["top_1.0pct_precision"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "cross_score_results.csv"), index=False)
    logger.info("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
