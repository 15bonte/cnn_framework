from pathlib import Path
import random
from datetime import datetime
import os


from ..dimensions import Dimensions
from ..tools import extract_patterns
from ...data.tools import get_folder_path


class DataSplit:
    """
    Split the dataset into train, validation and test.
    """

    def __init__(self, params, distinct_files):
        """
        distinct_files: All distinct files in the data folder. Names in file_name should be in this list.
        """
        self.data_sets = {
            "train": {
                "ratio": params.train_ratio,
                "number": params.train_number,
                "possible_files": [],
                "files": [],
                "file_name": params.train_file,
            },
            "val": {
                "ratio": params.val_ratio,
                "number": params.val_number,
                "possible_files": [],
                "files": [],
                "file_name": params.val_file,
            },
            "test": {
                "ratio": params.test_ratio,
                "number": params.test_number,
                "possible_files": [],
                "files": [],
                "file_name": params.test_file,
            },
        }

        for name, data_set in self.data_sets.items():
            # Either number or ratio must be provided, not both
            if data_set["ratio"] * data_set["number"] != 0:
                raise ValueError(
                    f"Both {name} ratio and number are not null, choice is ambiguous."
                )
            # If file name is provided, then possible elements are the ones written in the file
            current_filename = data_set["file_name"]
            if current_filename == "":
                continue
            if os.path.isfile(current_filename):
                with open(current_filename, "r") as f:
                    patterns = f.read().splitlines()
                data_set["possible_files"] = extract_patterns(distinct_files, patterns)
                # In this case, if neither ratio nor number is provided, then ratio is 100%
                if data_set["ratio"] == 0 and data_set["number"] == 0:
                    data_set["ratio"] = 1
            else:
                raise ValueError(f"{current_filename} does not exist.")

    @staticmethod
    def display_loading_source(name, data_set, dir_src):
        source_path = data_set["file_name"] if data_set["file_name"] != "" else dir_src
        number_ratio_message = "All elements"
        if data_set["ratio"] > 1:
            raise ValueError(f"{name} ratio is greater than 1.")
        if data_set["number"] == 0 and data_set["ratio"] == 0:
            print(f"No data is loaded for {name}")
        else:
            if data_set["ratio"] > 0:
                number_ratio_message = f"{int(data_set['ratio'] * 100)}% elements"
            elif data_set["number"] > 0:
                number_ratio_message = f"{data_set['number']} elements"
            print(f"{name} data is loaded from {source_path} - {number_ratio_message}")

    def generate_train_val_test_list(self, files, dir_src):
        print("### Data source ###")
        for name, data_set in self.data_sets.items():
            # Display information about the source of the data
            self.display_loading_source(name, data_set, dir_src)
            # If missing possible_files, consider the ones provided
            if len(data_set["possible_files"]) == 0:
                data_set["possible_files"] = files[:]
            # Replace ratio by number to only consider number in the following
            if data_set["ratio"] > 0:
                data_set["number"] = int(
                    data_set["ratio"] * len(data_set["possible_files"])
                )
        print("###################")

        # Return data set names, sorted by priority
        def sort_data_sets():
            # Train is always last as it uses random with no seed
            # Between test and val, first select the one having a file name
            if self.data_sets["test"]["file_name"] > self.data_sets["val"]["file_name"]:
                return ["test", "val", "train"]
            return ["val", "test", "train"]

        sorted_data_sets = sort_data_sets()
        for name in sorted_data_sets:
            data_set = self.data_sets[name]
            # Raise error if too many elements are required
            if data_set["number"] > len(data_set["possible_files"]):
                raise ValueError(f"Not enough files for {name}...")
            # Randomly sample indices
            # Keep same val and test images at each run
            rd_generator = random.Random(None if name == "train" else 10)
            indices = rd_generator.sample(
                range(len(data_set["possible_files"])), data_set["number"]
            )
            data_set["files"] = [data_set["possible_files"][i] for i in indices]
            # Remove the files from the possible files for other data sets
            for file in data_set["files"]:
                for other_data_set in self.data_sets.values():
                    if file in other_data_set["possible_files"]:
                        other_data_set["possible_files"].remove(file)

        # Sort files before returning
        return (
            sorted(self.data_sets["train"]["files"]),
            sorted(self.data_sets["val"]["files"]),
            sorted(self.data_sets["test"]["files"]),
        )


class BaseModelParams:
    """
    Model params base class.
    """

    def __init__(self, name):
        self.name = name

        # Input dimensions to the model
        self.input_dimensions = Dimensions(height=32, width=32)

        # Optimizer
        self.beta1 = 0.9  # exponential decay rate for the first moment estimates
        self.beta2 = 0.999  # exponential decay rate for the second moment estimates
        self.weight_decay = 0.0  # weight decay (L2 penalty)
        self.dropout = 0.0  # dropout rate

        # Training parameters
        self.num_epochs = 50
        self.batch_size = 32
        self.learning_rate = 0.1
        self.fp16_precision = False
        self.nb_warmup_epochs = 10
        self.num_workers = 4

        self.nb_classes = None
        self.class_names = []
        self.model_index = 0
        self.c_indexes = [0]  # channels selected as input
        self.z_indexes = [0]  # heights selected as input
        self.out_channels = 0  # output

        # Input data set
        self.data_dir = get_folder_path("images")

        # Data split
        self.train_ratio = 0
        self.train_number = 0
        self.train_file = ""

        self.val_ratio = 0
        self.val_number = 0
        self.val_file = ""

        self.test_ratio = 0
        self.test_number = 0
        self.test_file = ""

        self.cross_validation_dir = ""

        # Tensorboard parameters
        self.tensorboard_folder_path = get_folder_path("tensorboard")
        self.plot_step = 10
        # Number of different epochs where to plot images
        self.nb_plot_images = 2
        self.nb_tensorboard_images_max = 8

        # Output folders - models & predictions
        self.models_folder = get_folder_path("models")
        self.model_save_name = f"{self.name}.pt"
        self.output_dir = get_folder_path("predictions")

        # Path to load model to predict
        self.model_load_path = ""

        # Date
        self.format_now = ""
        self.job_id = "local"

        # Global results path
        self.global_results_path = ""

        # Files loaded - do not set, to be modified during data loading
        self.names_train = []
        self.names_val = []
        self.names_test = []

        # Weights and biases parameters
        self.wandb_project = "vae-dummy"
        self.wandb_entity = "cbio-bis"

    def get_useful_training_parameters(self):
        return f"epochs {self.num_epochs} | batch {self.batch_size} | lr {self.learning_rate} | weight decay {self.weight_decay} | dropout {self.dropout} | c {self.c_indexes} | z {self.z_indexes}"

    def update(self, args=None):
        if args is not None:
            if args.job_id:
                self.job_id = args.job_id
            if args.data_dir:
                self.data_dir = args.data_dir
            if args.tb_dir:
                self.tensorboard_folder_path = args.tb_dir
            if args.model_path:
                self.models_folder = args.model_path
            if args.lr:
                self.learning_rate = float(args.lr)
            if args.epochs:
                self.num_epochs = int(args.epochs)
            if args.plot_step:
                self.plot_step = int(args.plot_step)
            if args.output_dir:
                self.output_dir = args.output_dir
            if args.batch_size:
                self.batch_size = int(args.batch_size)
            if args.train_ratio:
                self.train_ratio = float(args.train_ratio)
            if args.val_ratio:
                self.val_ratio = float(args.val_ratio)
            if args.test_ratio:
                self.test_ratio = float(args.test_ratio)
            if args.train_number:
                self.train_number = int(args.train_number)
            if args.val_number:
                self.val_number = int(args.val_number)
            if args.test_number:
                self.test_number = int(args.test_number)
            if args.train_file:
                self.train_file = args.train_file
            if args.val_file:
                self.val_file = args.val_file
            if args.test_file:
                self.test_file = args.test_file
            if args.model_load_path:
                self.model_load_path = args.model_load_path
            if args.global_results_path:
                self.global_results_path = args.global_results_path
            if args.model_index:
                self.model_index = int(args.model_index)
            if args.image_height:
                self.input_dimensions.height = int(args.image_height)
            if args.image_width:
                self.input_dimensions.width = int(args.image_width)
            if args.weight_decay:
                self.weight_decay = float(args.weight_decay)
            if args.num_workers:
                self.num_workers = int(args.num_workers)
            if args.dropout:
                self.dropout = float(args.dropout)
            if args.c_indexes:
                self.c_indexes = args.c_indexes
            if args.z_indexes:
                self.z_indexes = args.z_indexes
            if args.cross_validation_dir:
                self.cross_validation_dir = args.cross_validation_dir
            if args.out_channels:
                self.out_channels = int(args.out_channels)

        # Create folders dedicated to current run
        now = datetime.now()
        self.format_now = now.strftime("%Y%m%d-%H%M%S") + "-" + self.job_id

        print(f"Model time id: {self.format_now}")
        print(self.get_useful_training_parameters())

        self.tensorboard_folder_path = (
            f"{self.tensorboard_folder_path}/{self.format_now}_{self.name}"
        )

        self.models_folder = f"{self.models_folder}/{self.name}/{self.format_now}"

        self.output_dir = f"{self.output_dir}/{self.name}/{self.format_now}"
