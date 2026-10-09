#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

"""Convert the efficientnet_lite_pytorch ImageNet init into timm key format.

Produces a timm-keyed state dict of the *pretrained* (untrained) lite0 backbone —
the proven-good FixMatch init — so timm's ``efficientnet_lite0`` architecture can
be initialised with it without the `efficientnet_lite_pytorch` runtime dependency.
Saved as safetensors next to this script's results dir.
"""

import os

import efficientnet_lite_pytorch
import timm
from efficientnet_lite0_pytorch_model import EfficientnetLite0ModelFile
from loguru import logger
from safetensors.torch import save_file

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "converted_init")


def main():
    """Map v1 pretrained lite0 weights onto timm keys and save.

    Raises:
        ValueError: If a v1 tensor and its positionally-mapped timm key differ in shape.
    """
    os.makedirs(OUT, exist_ok=True)
    # num_classes=1000 so the classifier matches the pretrained head; we drop the
    # classifier when initialising a 2-class model, so only the backbone matters.
    v1 = efficientnet_lite_pytorch.EfficientNet.from_pretrained(
        "efficientnet-lite0",
        weights_path=EfficientnetLite0ModelFile.get_model_file_path(),
        num_classes=1000,
        in_channels=3,
    )
    tm = timm.create_model(
        "tf_efficientnet_lite0.in1k", pretrained=False, num_classes=1000, in_chans=3
    )

    timm_keys = list(tm.state_dict().keys())
    v1_items = list(v1.state_dict().items())
    assert len(timm_keys) == len(v1_items), (len(timm_keys), len(v1_items))
    mapped = {}
    for tk, (vk, vt) in zip(timm_keys, v1_items):
        ref = tm.state_dict()[tk]
        if tuple(ref.shape) != tuple(vt.shape):
            raise ValueError(f"shape mismatch {tk}{tuple(ref.shape)} vs {vk}{tuple(vt.shape)}")
        mapped[tk] = vt.contiguous()

    # Verify it loads cleanly into the timm arch.
    missing, unexpected = tm.load_state_dict(mapped, strict=True)
    logger.info(f"verify load into timm: missing={len(missing)} unexpected={len(unexpected)}")

    path = os.path.join(OUT, "efficientnet_lite0_v1init_timmkeys.safetensors")
    save_file(mapped, path)
    logger.info(f"saved converted init: {path} ({len(mapped)} tensors)")
    print(path)


if __name__ == "__main__":
    main()
