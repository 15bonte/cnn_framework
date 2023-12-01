from cnn_framework.dummy_regression_cnn.model_params import DummyModelParams
from cnn_framework.dummy_regression_cnn.train import training


def test_training():
    """
    Test dummy regression.
    """
    params = DummyModelParams()
    params.update()

    score = training(params)
    assert abs(score) < 30
