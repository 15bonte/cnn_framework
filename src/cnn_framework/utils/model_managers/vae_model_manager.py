import os
from matplotlib import pyplot as plt

from ..enum import PredictMode
from ..data_sets.dataset_output import DatasetOutput
from ..display_tools import display_progress
from .model_manager import ModelManager


class VAEModelManager(ModelManager):
    def compute_loss(self, dl_element, dl_metric, _=None, __=None):
        # Read data loader element
        dl_element["data"] = dl_element["data"].to(self.device)
        dl_element["category"] = dl_element["category"].to(self.device)

        # Compute the model output
        model_output = self.model(dl_element)
        dl_element.prediction = model_output["recon_x"]

        # Update metric
        dl_metric.update(model_output["recon_x"], dl_element["data"])

        return model_output["loss"]

    def model_prediction(self, dl_element, dl_metric, _):
        """
        Function to generate outputs from inputs for given model.
        """
        # Read data loader element
        dl_element["data"] = dl_element["data"].to(self.device)
        dl_element["category"] = dl_element["category"].to(self.device)

        # Compute the model output
        model_output = self.model(dl_element)
        dl_element.prediction = model_output

        # Update metric
        dl_metric.update(model_output["recon_x"], dl_element["data"])

    def get_embedding(self, dl_element):
        dl_element["data"] = dl_element["data"].to(self.device)
        return self.model.encoder(dl_element["data"]).embedding

    def batch_predict(
        self, test_dl, images_to_save, num_batches_test, test_metric, predict_mode: PredictMode
    ):
        all_predictions_np = []
        for batch_idx, dl_element in enumerate(test_dl):
            # Run prediction
            if predict_mode == PredictMode.Standard:  # standard use
                self.model_prediction(dl_element, test_metric, test_dl)
                predictions = dl_element.prediction["recon_x"]
            else:  # return embedding
                predictions = self.get_embedding(dl_element)

            predictions_np = predictions.cpu().numpy()
            all_predictions_np = all_predictions_np + [*predictions_np]

            display_progress(
                "Model evaluation in progress",
                batch_idx + 1,
                num_batches_test,
                additional_message=f"Batch #{batch_idx}",
            )

            # Save few images
            if predict_mode != PredictMode.Standard:
                continue

            # Get numpy elements
            inputs_np = dl_element["data"].cpu().numpy()

            for idx in range(dl_element["data"].shape[0]):
                if self.image_index in images_to_save:
                    image_id = (batch_idx * test_dl.batch_size) + idx
                    image_name = test_dl.dataset.names[image_id].split(".")[0]

                    # Get element at index
                    input_np = inputs_np[idx, ...].squeeze()
                    prediction_np = predictions_np[idx, ...].squeeze()

                    dl_element_numpy = DatasetOutput(
                        input=input_np,
                        prediction=prediction_np,
                    )

                    self.save_results(
                        f"{image_name}_{self.image_index}",
                        dl_element_numpy,
                        test_dl.dataset.mean_std,
                    )

                self.image_index += 1

        return all_predictions_np

    def write_useful_information(self):
        # Update parameters file with all useful information
        os.makedirs(self.params.models_folder, exist_ok=True)
        with open(self.parameters_path, "a") as f:
            for attribute, value in vars(self.training_information).items():
                f.write("%s;%s\n" % (attribute, value))
        f.close()

        if self.params.global_results_path == "":
            return

        # If global results file does not exist, create it
        if not os.path.exists(self.params.global_results_path):
            with open(self.params.global_results_path, "w") as f:
                f.write(
                    "data;id;latent dim;learning rate;beta;gamma;delta;git hash;score;weight_decay;batch size;drop out;encoder name;additional score\n"
                )
            f.close()

        # Store useful results in global results file
        with open(self.params.global_results_path, "a") as f:
            f.write(f"{self.params.data_dir};")
            f.write(f"{self.params.format_now};")
            f.write(f"{self.params.latent_dim};")
            f.write(f"{self.params.learning_rate};")
            f.write(f"{self.params.beta};")
            f.write(f"{self.params.gamma};")
            f.write(f"{self.params.delta};")
            f.write(f"{self.training_information.git_hash};")
            f.write(f"{self.training_information.score};")
            f.write(f"{self.params.weight_decay};")
            f.write(f"{self.params.batch_size};")
            f.write(f"{self.params.dropout};")
            f.write(f"{self.params.encoder_name};")
            f.write(f"{self.training_information.additional_score};\n")
        f.close()

    def write_images_to_tensorboard(self, current_batch, dl_element, name):
        # Get numpy arrays
        inputs_np = dl_element["data"]
        image_indexes_np = dl_element["id"]

        # Get images name
        current_dl_file_names = [
            file_name.split(".")[0] for file_name in self.dl[name].dataset.names
        ]
        image_names = [current_dl_file_names[image_index] for image_index in image_indexes_np]

        # Log the results images
        for i, (input_np, image_name) in enumerate(zip(inputs_np, image_names)):
            # Do not save too many images
            if i == self.params.nb_tensorboard_images_max:
                break
            for channel in range(input_np.shape[0]):
                # ... log the ground truth image
                plt.imshow(input_np[channel], cmap="gray")
                self.writer.add_figure(
                    f"{name}/{image_name}/{channel}/input",
                    plt.gcf(),
                    current_batch,
                )
