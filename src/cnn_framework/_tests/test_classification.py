import numpy as np

from cnn_framework.dummy_cnn.model_params import DummyModelParams
from cnn_framework.dummy_cnn.train import training


def test_training():
    """
    Test dummy classification.
    """
    params = DummyModelParams()
    params.update()

    score = training(params)
    assert np.isclose(score, 1.0, rtol=5e-02)
