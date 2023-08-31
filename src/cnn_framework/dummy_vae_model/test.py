import os
import numpy as np
from pythae.models import AutoModel
import umap
import plotly.express as px
from sklearn.linear_model import LogisticRegression

from .data_set import DummyVAEDataSet
from .model_params import DummyVAEModelParams

from ..utils.DataManagers import DefaultDataManager
from ..utils.data_loader_generators.DataLoaderGenerator import DataLoaderGenerator
from ..utils.metrics import PCC
from ..utils.model_managers.VAEModelManager import VAEModelManager
from ..utils.parsers.VAEParser import VAEParser


def main(params):
    loader_generator = DataLoaderGenerator(params, DummyVAEDataSet, DefaultDataManager)
    train_dl, _, test_dl = loader_generator.generate_data_loader(shuffle_train=False)

    model = AutoModel.load_from_folder(params.model_load_path)

    manager = VAEModelManager(model, params, PCC)

    # manager.predict(test_dl)

    # Run predictions on train and test
    predictions_train = np.array(manager.predict(train_dl, return_predictions=True))
    classes_train = np.array(
        [
            train_dl.dataset.names[idx]
            .split(".")[0]
            .split("_c")[1]  # to be checked for all data sets
            for dl_element in train_dl
            for idx in dl_element["id"].detach().numpy()
        ]
    )

    predictions_test = np.array(manager.predict(test_dl, return_predictions=True))
    classes_test = np.array(
        [
            test_dl.dataset.names[idx]
            .split(".")[0]
            .split("_c")[1]  # to be checked for all data sets
            for dl_element in test_dl
            for idx in dl_element["id"].detach().numpy()
        ]
    )

    # Perform UMAP on test
    results = umap.UMAP().fit_transform(predictions_test)
    fig = px.scatter(
        results,
        x=0,
        y=1,
        color=classes_test,
        labels={"color": "class"},
        color_discrete_sequence=px.colors.qualitative.G10,
    )
    fig.write_html(os.path.join(params.output_dir, "umap.html"))

    # Learn logistic regression on train
    clf = LogisticRegression(random_state=0).fit(predictions_train, classes_train)
    # Print score on train & test
    print(f"Logistic classifier on train: {clf.score(predictions_train, classes_train)}")
    print(f"Logistic classifier on test: {clf.score(predictions_test, classes_test)}")


if __name__ == "__main__":
    parser = VAEParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyVAEModelParams()
    parameters.update(args)

    main(parameters)
