#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Generate the shared efficientnet-lite0 checkpoint used across the test suite.

The checkpoint is a 26 MB binary that used to be committed to the repository.
Because every revision of it stayed in the pack forever, it dominated clone
size, so it is generated on demand instead and git-ignored.

Weights are randomly initialised (``pretrained=False``): no test asserts on
score *values* — the e2e prediction tests seed their own sentinel scores — so
only the checkpoint's structure matters. This also keeps the generator free of
any network access, which matters in CI.

Because the checkpoint is now generated from the *current* architecture, it can
no longer catch an architecture change on its own — writer and reader always
agree. ``STATE_DICT_MANIFEST_PATH`` is what preserves that regression: a small
committed file pinning every ``state_dict`` key and its shape, which
``TestStoredModelLoading`` compares the generated checkpoint against. Refresh it
deliberately (``--update-manifest``) when the architecture is meant to change.

Usage:
    python tests/test_data/generate_test_model.py [--update-manifest]
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MODEL_PATH = SCRIPT_DIR / "test_model.safetensors"
STATE_DICT_MANIFEST_PATH = SCRIPT_DIR / "test_model_state_dict.json"

# Generators inside tests/test_data are run as standalone scripts, so the
# repository root is not on sys.path yet.
sys.path.insert(0, str(REPO_ROOT))

# Fixed so every checkout produces byte-identical weights: a test that fails
# against this checkpoint must be reproducible on someone else's machine.
SEED = 42

# Module-level imports are stdlib only, and torch/timm/fitsbolt are pulled in
# inside the functions that need them: ``tests/conftest.py`` imports this module
# at collection time, and a top-level ``import torch`` there would cost every
# xdist worker several seconds whether or not it ever builds the checkpoint.


def generate_test_model(model_path: Path = MODEL_PATH) -> Path:
    """Write an efficientnet-lite0 checkpoint in ``SessionIOHandler.save_model`` format.

    Args:
        model_path: Destination for the checkpoint.

    Returns:
        Path: The written checkpoint path.
    """
    from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

    from anomaly_match.data_io.checkpoint_io import save_checkpoint
    from anomaly_match.data_io.load_images import get_fitsbolt_config
    from anomaly_match.models.FixMatch import FixMatch
    from anomaly_match.utils.get_default_cfg import get_default_cfg
    from anomaly_match.utils.get_net_builder import get_net_builder
    from anomaly_match.utils.set_seeds import set_seeds

    set_seeds(SEED)

    cfg = get_default_cfg()
    cfg.net = "efficientnet-lite0"
    cfg.pretrained = False
    cfg.num_channels = 3
    cfg.num_workers = 0
    cfg.normalisation.image_size = [150, 150]
    cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    cfg.normalisation.n_output_channels = 3
    cfg = get_fitsbolt_config(cfg)

    net_builder = get_net_builder(cfg.net, pretrained=cfg.pretrained, in_channels=cfg.num_channels)
    model = FixMatch(
        net_builder,
        num_classes=2,
        in_channels=cfg.num_channels,
        ema_m=cfg.ema_m,
        T=1.0,
        p_cutoff=cfg.p_cutoff,
        lambda_u=cfg.ulb_loss_ratio,
    )

    save_state = {
        "train_model": model.train_model.state_dict(),
        "eval_model": model.eval_model.state_dict(),
        "optimizer": model.optimizer.state_dict() if model.optimizer else None,
        "scheduler": model.scheduler.state_dict() if model.scheduler else None,
        "it": 0,
        "total_it": 0,
        "best_eval_acc": 0.0,
        "best_it": 0,
        "last_normalisation_method": NormalisationMethod.CONVERSION_ONLY,
        "normalisation_method": NormalisationMethod.CONVERSION_ONLY,
        "num_channels": cfg.num_channels,
        "net": cfg.net,
        "fitsbolt_cfg": cfg.fitsbolt_cfg,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place: a half-written checkpoint (Ctrl+C, a
    # CI timeout, two pytest processes racing) would satisfy every later
    # ``exists()`` check and then fail as an opaque safetensors error.
    # ``save_checkpoint`` forces a ``.safetensors`` suffix, so the staged name
    # carries it and the written path comes back from the call.
    staged_path = model_path.with_name(f".{model_path.stem}.{os.getpid()}.safetensors")
    try:
        written_path = save_checkpoint(save_state, staged_path)
        os.replace(written_path, model_path)
    finally:
        staged_path.unlink(missing_ok=True)
    return model_path


def ensure_test_model(model_path: Path = MODEL_PATH) -> Path:
    """Return the shared checkpoint, generating it if it is not on disk.

    The single entry point for every consumer (the ``test_model_path`` fixture,
    ``tests/demo_profiler.py``, ``scripts/validate_browser_testing.py``) so that
    "build it if missing" is defined once.

    Args:
        model_path: Checkpoint location to check and, if needed, write.

    Returns:
        Path: The checkpoint path, now guaranteed to exist.
    """
    if not model_path.exists():
        print(f"Generating {model_path.name} (one-off, ~26 MB)...")
        generate_test_model(model_path)
    return model_path


def state_dict_manifest(model_path: Path = MODEL_PATH) -> dict:
    """Map every ``state_dict`` key in a checkpoint to its tensor shape.

    Args:
        model_path: Checkpoint to read.

    Returns:
        dict: ``{"train_model": {key: [dims]}, "eval_model": {...}}``.
    """
    from anomaly_match.data_io.checkpoint_io import load_checkpoint

    checkpoint = load_checkpoint(model_path)
    return {
        part: {key: list(tensor.shape) for key, tensor in sorted(checkpoint[part].items())}
        for part in ("train_model", "eval_model")
    }


def main():
    """Generate the checkpoint, optionally refreshing the pinned key manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Rewrite test_model_state_dict.json from the generated checkpoint. "
        "Only do this when the model architecture is meant to have changed.",
    )
    args = parser.parse_args()

    print(f"Creating {MODEL_PATH.name}...")
    path = generate_test_model()
    print(f"  Saved {path.relative_to(REPO_ROOT)} ({path.stat().st_size / (1024 * 1024):.1f} MB)")

    if args.update_manifest:
        STATE_DICT_MANIFEST_PATH.write_text(
            json.dumps(state_dict_manifest(path), indent=1) + "\n", encoding="utf-8"
        )
        print(f"  Updated {STATE_DICT_MANIFEST_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
