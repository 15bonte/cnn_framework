import torch


class FileNameEncoder:
    """
    Use to convert file name to int, so that it can be included into DatasetOutput.
    """

    def __init__(
        self,
        names_train: list[str],
        names_val: list[str],
        names_test: list[str],
    ):
        self.names_train = names_train
        self.names_val = names_val
        self.names_test = names_test

    def encode(self, s: str) -> int:
        try:
            train_idx = self.names_train.index(s)
            return int(f"1{train_idx}")
        except ValueError:
            try:
                val_idx = self.names_val.index(s)
                return int(f"2{val_idx}")
            except ValueError:
                test_idx = self.names_test.index(s)
                return int(f"3{test_idx}")

    def decode(self, i: int) -> str:
        if torch.is_tensor(i):
            i = i.item()
        if str(i)[0] == "1":
            return self.names_train[int(str(i)[1:])]
        if str(i)[0] == "2":
            return self.names_val[int(str(i)[1:])]
        return self.names_test[int(str(i)[1:])]
