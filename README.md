# CNN framework

[![codecov](https://codecov.io/gh/15bonte/cnn_framework/branch/main/graph/badge.svg)](https://codecov.io/gh/15bonte/cnn_framework)
[![tests](https://github.com/15bonte/cnn_framework/workflows/tests/badge.svg)](https://github.com/15bonte/cnn_framework/actions)

Run CNN models for classification, regression, segmentation, VAE, contrastive learning with any data set.

## Installation

First, create a dedicated conda environment using Python 3.9

```bash
conda create -n cnn_framework python=3.9
conda activate cnn_framework
```

To install the latest github version of this library run the following using pip

```bash
pip install git+https://github.com/15bonte/cnn_framework
```

or alternatively you can clone the github repository

```bash
git clone https://github.com/15bonte/cnn_framework.git
cd cnn_framework
pip install -e .
```

Independently install pytorch-related packages. Note that version are specified as this repository is not really maintained. Could work with later versions, though. `setuptools` version is specified as `torchmetrics` needs this old version.

```bash
pip install torch==1.12.1 torchvision==0.13.1 torchmetrics==0.11.4 segmentation-models-pytorch==0.3.0 setuptools==80.9.0
```

If you want to run jupyter tutorials, you also need to install ipykernel

```bash
pip install ipykernel
```

If you want to work with VAE, you must also install [Pythae](https://github.com/clementchadebec/benchmark_VAE/tree/main) and [WandB](https://wandb.ai/home), which is not the case by default.

```bash
pip install pythae
```

```bash
pip install wandb
```
