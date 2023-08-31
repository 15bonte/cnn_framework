from argparse import ArgumentParser


class PasteurParser(ArgumentParser):
    """
    Parser class for all Pasteur scripts.
    """

    def __init__(self):
        super().__init__()

        self.add_argument("--video_folder", help="Path to folder containing original videos.")
        self.add_argument(
            "--xml_models_dir", help="Path to folder containing xml models created by Trackmate."
        )

        self.add_argument("--metaphase_model", help="Path to CNN metaphase model.")
        self.add_argument(
            "--predictions_file",
            help="Path to save metaphase predictions for HMM optimization.",
            default=None,
        )
        self.add_argument(
            "--mitosis_ground_truth_folder",
            help="Path to mitosis ground truth folder.",
            default=None,
        )
        self.add_argument(
            "--annotations_folder",
            help="Path to folder containing annotations.",
            default=None,
        )
        self.add_argument("--hmm_parameters_file", help="Path to metaphase HMM parameters file.")
        self.add_argument("--mitoses_save_dir", help="Path to mitoses bin save directory.")
        self.add_argument("--tracks_save_dir", help="Path to tracks bin save directory.")
        self.add_argument("--movies_save_dir", help="Path to mitosis movies save directory.")
        self.add_argument("--save_movies", action="store_true", help="Save mitosis movies.")
        self.add_argument("--no-save_movies", dest="save_movies", action="store_false")
        self.set_defaults(save_movies=True)

        self.add_argument("--bridges_scaler_path", help="Path to bridges scaler file.")
        self.add_argument("--bridges_model_path", help="Path to bridges model file.")

        self.add_argument("--bridge_images_path", help="Path to bridge images.")
