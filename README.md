# CNN framework

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

If you want to work with VAE, you must also install [Pythae](https://github.com/clementchadebec/benchmark_VAE/tree/main), which is not the case by default.

```bash
pip install pythae
```
