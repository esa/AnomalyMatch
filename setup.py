#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
from setuptools import setup, find_packages

setup(
    name="anomaly_match",
    version="1.0.0",
    description="A tool for anomaly detection in images using semi-supervised and active learning with a GUI",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "albumentations",
        "dotmap",
        "efficientnet-pytorch",
        "efficientnet_lite_pytorch",
        "efficientnet_lite0_pytorch_model",
        "h5py",
        "ipykernel",
        "ipywidgets",
        "imageio",
        "loguru",
        "matplotlib",
        "numpy",
        "opencv-python-headless",
        "pandas",
        "pyturbojpeg",
        "scikit-learn",
        "scikit-image",
        "toml",
        "torch",
        "torchvision",
        "tqdm",
    ],
)
