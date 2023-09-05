from .cnn_parser import CnnParser


class TransferParser(CnnParser):
    """
    Transfer learning parsing class.
    """

    def __init__(self):
        super().__init__()

        self.arguments_parser.add_argument(
            "--model_pretrained_path", help="Path to segmentation pretrained model"
        )
