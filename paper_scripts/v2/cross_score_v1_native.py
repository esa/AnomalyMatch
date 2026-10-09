#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Variance-immune test: score v1's own trained model via v2's preprocessing.

Builds the v1 architecture (``efficientnet_lite_pytorch``), loads v1's saved
``eval_model`` weights (the model that reported AUROC 0.961), and scores all
65k images through v2's fitsbolt preprocessing + [0,1] tensor conversion — the
exact pixel pipeline v2 prediction uses. No training happens, so the result has
no run-to-run variance.

- Result ≈ 0.96 → v2's scoring/preprocessing faithfully scores a good model, so
  the v2 shortfall is in **training** (and, given the large cold-start variance,
  is partly noise / a training-dynamics difference).
- Result ≈ 0.81 → v2's preprocessing degrades even a known-good model → a real
  **scoring/preprocessing** regression.

Run under the ``am`` env. Independent of the current ``transforms.py`` patch
(applies its own [0,1] ToTensor).
"""

import os
import sys
import types

import numpy as np
import torch
from loguru import logger

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Compat alias so torch.load can unpickle v1's checkpoint (moved enum).
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod  # noqa: E402

_shim = types.ModuleType("anomaly_match.image_processing.NormalisationMethod")
_shim.NormalisationMethod = NormalisationMethod
sys.modules["anomaly_match.image_processing.NormalisationMethod"] = _shim

import benchmark_datasets as bd  # noqa: E402
import efficientnet_lite_pytorch  # noqa: E402
from benchmark_metrics import evaluate_performance  # noqa: E402
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile  # noqa: E402

# fitsbolt batch loader (multi-file decode).
from fitsbolt.image_loader import load_and_process_images  # noqa: E402
from run_benchmark import list_image_files  # noqa: E402

import anomaly_match as am  # noqa: E402
from anomaly_match.data_io.load_images import get_fitsbolt_config  # noqa: E402

V1_MODELS = {
    "iter1(cold,0.9614)": (
        "/media/team_workspaces/AnomalyMatch/paper_results/FullCorrectedOutput/miniimagenet/"
        "mini_class57_ratio0.01/miniimagenet_anomaly57_n495_a5/"
        "benchmark_miniimagenet_anomaly57_20250726_173340/model_iteration_0.pth"
    ),
    "iter3(final,0.9702)": (
        "/media/team_workspaces/AnomalyMatch/paper_results/FullCorrectedOutput/miniimagenet/"
        "mini_class57_ratio0.01/miniimagenet_anomaly57_n495_a5/"
        "benchmark_miniimagenet_anomaly57_20250726_173340/model_iteration_2.pth"
    ),
}


def build_v1_model(pth, device):
    """Build the v1 efficientnet-lite0 and load a saved eval_model state dict.

    Returns:
        The eval-mode model on ``device`` with v1's trained weights loaded.
    """
    weights_path = EfficientnetLite0ModelFile.get_model_file_path()
    model = efficientnet_lite_pytorch.EfficientNet.from_pretrained(
        "efficientnet-lite0", weights_path=weights_path, num_classes=2, in_channels=3
    )
    state = torch.load(pth, weights_only=False, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["eval_model"], strict=True)
    logger.info(
        f"Loaded {os.path.basename(pth)}: missing={len(missing)} unexpected={len(unexpected)}"
    )
    return model.to(device).eval()


def score_all(model, files, fitsbolt_cfg, device, batch_size=512):
    """Score every file: fitsbolt load -> [0,1] tensor -> softmax anomaly prob.

    Returns:
        A float32 array of per-file anomaly probabilities, aligned with ``files``.
    """
    scores = np.empty(len(files), dtype=np.float32)
    for start in range(0, len(files), batch_size):
        chunk = files[start : start + batch_size]
        imgs = load_and_process_images(chunk, cfg=fitsbolt_cfg, show_progress=False)
        # imgs: list/array of HWC uint8. ToTensor-equivalent: CHW float /255 -> [0,1].
        batch = np.stack(imgs, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)
        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.nn.functional.softmax(logits.float(), dim=-1)[:, 1]
        scores[start : start + len(chunk)] = probs.cpu().numpy()
        if (start // batch_size) % 20 == 0:
            logger.info(f"  scored {start + len(chunk)}/{len(files)}")
    return scores


def main():
    """Score v1 models via v2 preprocessing and report AUROC per model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = bd.REGISTRY["miniimagenet"]
    anomaly_idx = bd.resolve_anomaly_class(dataset, "hourglass")
    gt_df = bd.load_ground_truth(dataset)
    full_paths = list_image_files(dataset.image_dir)  # absolute paths
    basenames = [os.path.basename(p) for p in full_paths]  # match gt_df.filename

    # Build the same fitsbolt preprocessing v2 uses (CONVERSION_ONLY, 224, order 1).
    cfg = am.get_default_cfg()
    cfg.data_dir = dataset.image_dir
    cfg.normalisation.image_size = [224, 224]
    cfg.normalisation.n_output_channels = 3
    cfg = get_fitsbolt_config(cfg, size_override="default")
    fitsbolt_cfg = cfg.fitsbolt_cfg

    for tag, pth in V1_MODELS.items():
        logger.info(f"===== {tag} =====")
        model = build_v1_model(pth, device)
        scores = score_all(model, full_paths, fitsbolt_cfg, device)
        metrics = evaluate_performance(scores, basenames, gt_df, anomaly_idx)
        logger.info(
            f"[{tag}] v1-model-via-v2-preproc: AUROC={metrics['auroc']:.4f} "
            f"AUPRC={metrics['auprc']:.4f} top-1%={metrics['top_1.0pct_precision']:.1f}%"
        )


if __name__ == "__main__":
    main()
