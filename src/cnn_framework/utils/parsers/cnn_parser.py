import argparse


class CnnParser:
    """
    CNN parsing class.
    """

    def __init__(self):
        self.arguments_parser = argparse.ArgumentParser()

        self.arguments_parser.add_argument(
            "--data_dir", help="Folder containing .ciz pictures"
        )
        self.arguments_parser.add_argument("--tb_dir", help="Tensorboard folder path")
        self.arguments_parser.add_argument("--model_path", help="Model save path")
        self.arguments_parser.add_argument("--lr", help="Learning rate")
        self.arguments_parser.add_argument("--epochs", help="Number of epochs")
        self.arguments_parser.add_argument("--plot_step", help="Plot every n steps")
        self.arguments_parser.add_argument(
            "--output_dir", help="Folder to save output pictures"
        )
        self.arguments_parser.add_argument(
            "--train_ratio", help="Ratio of input pictures to include in training set"
        )
        self.arguments_parser.add_argument(
            "--val_ratio", help="Ratio of input pictures to include in validation set"
        )
        self.arguments_parser.add_argument(
            "--test_ratio", help="Ratio of input pictures to include in test set"
        )
        self.arguments_parser.add_argument(
            "--train_number", help="Number of input pictures to include in training set"
        )
        self.arguments_parser.add_argument(
            "--val_number", help="Number of input pictures to include in validation set"
        )
        self.arguments_parser.add_argument(
            "--test_number", help="Number of input pictures to include in test set"
        )
        self.arguments_parser.add_argument(
            "--train_file", help="File containing training set"
        )
        self.arguments_parser.add_argument(
            "--val_file", help="File containing validation set"
        )
        self.arguments_parser.add_argument(
            "--test_file", help="File containing test set"
        )
        self.arguments_parser.add_argument(
            "--cross_validation_dir", help="Folder containing cross validation splits"
        )
        self.arguments_parser.add_argument("--batch_size", help="Batch size")
        self.arguments_parser.add_argument(
            "--model_load_path", help="Model load path to predict"
        )
        self.arguments_parser.add_argument(
            "--global_results_path", help="Path with all results"
        )
        self.arguments_parser.add_argument("--job_id", help="Slurm job id")
        self.arguments_parser.add_argument(
            "--model_index", help="Potential index to choose model to be used"
        )
        self.arguments_parser.add_argument("--image_height", help="Image height")
        self.arguments_parser.add_argument("--image_width", help="Image width")
        self.arguments_parser.add_argument("--weight_decay", help="Weight decay")
        self.arguments_parser.add_argument("--dropout", help="Dropout rate")
        self.arguments_parser.add_argument(
            "--num_workers", help="Number of workers in train/val data loaders"
        )
        self.arguments_parser.add_argument(
            "--c_indexes",
            nargs="*",
            type=int,
            help="Channel indexes to select as input",
        )
        self.arguments_parser.add_argument(
            "--z_indexes", nargs="*", type=int, help="Height indexes to select as input"
        )
        self.arguments_parser.add_argument(
            "--out_channels", help="Number of channels as output"
        )
