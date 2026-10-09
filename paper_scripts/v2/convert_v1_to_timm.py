#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Weights-vs-architecture test: run v1's trained weights on timm's architecture.

The v1 (`efficientnet_lite_pytorch`) and timm (`efficientnet_lite0`) state dicts
align positionally (296 keys, identical shape sequence, distinct per-layer shapes
→ unambiguous mapping). This loads v1's *trained* 0.967 hourglass model into the
timm architecture and scores it through v2's [0,1] preprocessing.

- ≈ 0.967 → the timm architecture is numerically equivalent; the regression is
  purely the timm **pretrained init weights** (next: convert v1's ImageNet
  pretrained weights into timm and train).
- ≪ 0.967 → the timm architecture itself is not equivalent.
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

from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod  # noqa: E402

_shim = types.ModuleType("anomaly_match.image_processing.NormalisationMethod")
_shim.NormalisationMethod = NormalisationMethod
sys.modules["anomaly_match.image_processing.NormalisationMethod"] = _shim

import benchmark_datasets as bd  # noqa: E402
import efficientnet_lite_pytorch  # noqa: E402
import timm  # noqa: E402
from benchmark_metrics import evaluate_performance  # noqa: E402
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile  # noqa: E402
from fitsbolt.image_loader import load_and_process_images  # noqa: E402
from run_benchmark import list_image_files  # noqa: E402

import anomaly_match as am  # noqa: E402
from anomaly_match.data_io.load_images import get_fitsbolt_config  # noqa: E402

V1_TRAINED = (
    "/media/team_workspaces/AnomalyMatch/paper_results/FullCorrectedOutput/miniimagenet/"
    "mini_class57_ratio0.01/miniimagenet_anomaly57_n495_a5/"
    "benchmark_miniimagenet_anomaly57_20250726_173340/model_iteration_0.pth"
)


def map_v1_into_timm(v1_state_dict, timm_model):
    """Positionally map a v1 lite0 state dict onto a timm efficientnet_lite0.

    Args:
        v1_state_dict: state dict from ``efficientnet_lite_pytorch`` lite0.
        timm_model: an instantiated timm ``efficientnet_lite0`` model.

    Returns:
        A new state dict keyed by timm names with v1 tensors.

    Raises:
        ValueError: If any positionally-paired tensors differ in shape.
    """
    timm_keys = list(timm_model.state_dict().keys())
    v1_items = list(v1_state_dict.items())
    if len(timm_keys) != len(v1_items):
        raise ValueError(f"key count differs: timm {len(timm_keys)} vs v1 {len(v1_items)}")
    out = {}
    for tk, (vk, vt) in zip(timm_keys, v1_items):
        ts = timm_model.state_dict()[tk].shape
        if tuple(ts) != tuple(vt.shape):
            raise ValueError(f"shape mismatch {tk}{tuple(ts)} vs {vk}{tuple(vt.shape)}")
        out[tk] = vt
    return out


def main():
    """Convert v1's trained model into timm and score via v2 preprocessing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = bd.REGISTRY["miniimagenet"]
    anomaly_idx = bd.resolve_anomaly_class(dataset, "hourglass")
    gt_df = bd.load_ground_truth(dataset)
    full_paths = list_image_files(dataset.image_dir)
    basenames = [os.path.basename(p) for p in full_paths]

    # v1 trained weights
    v1 = efficientnet_lite_pytorch.EfficientNet.from_pretrained(
        "efficientnet-lite0",
        weights_path=EfficientnetLite0ModelFile.get_model_file_path(),
        num_classes=2,
        in_channels=3,
    )
    state = torch.load(V1_TRAINED, weights_only=False, map_location="cpu")
    v1.load_state_dict(state["eval_model"], strict=True)

    # timm arch, load v1 trained weights positionally
    tm = timm.create_model(
        "tf_efficientnet_lite0.in1k", pretrained=False, num_classes=2, in_chans=3
    )
    mapped = map_v1_into_timm(v1.state_dict(), tm)
    missing, unexpected = tm.load_state_dict(mapped, strict=True)
    logger.info(f"loaded v1->timm: missing={len(missing)} unexpected={len(unexpected)}")
    tm = tm.to(device).eval()

    # v2 preprocessing (CONVERSION_ONLY, 224, [0,1]); v1 weights were trained on [0,1]
    cfg = am.get_default_cfg()
    cfg.data_dir = dataset.image_dir
    cfg.normalisation.image_size = [224, 224]
    cfg.normalisation.n_output_channels = 3
    cfg = get_fitsbolt_config(cfg, size_override="default")
    fitsbolt_cfg = cfg.fitsbolt_cfg

    scores = np.empty(len(full_paths), dtype=np.float32)
    bs = 512
    for start in range(0, len(full_paths), bs):
        chunk = full_paths[start : start + bs]
        imgs = load_and_process_images(chunk, cfg=fitsbolt_cfg, show_progress=False)
        batch = np.stack(imgs, axis=0).astype(np.float32) / 255.0
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device)
        with torch.inference_mode():
            p = torch.nn.functional.softmax(tm(t).float(), dim=-1)[:, 1]
        scores[start : start + len(chunk)] = p.cpu().numpy()

    metrics = evaluate_performance(scores, basenames, gt_df, anomaly_idx)
    logger.info(
        f"[v1-weights on timm-arch] AUROC={metrics['auroc']:.4f} "
        f"AUPRC={metrics['auprc']:.4f} top-1%={metrics['top_1.0pct_precision']:.1f}%  "
        f"(v1 native = 0.9617)"
    )


if __name__ == "__main__":
    main()
