from pathlib import Path


class TrainingInformation:
    """
    Class to store training information.
    """

    def __init__(self, num_epochs: int) -> None:
        # Global parameters
        self.num_batches_train = None
        self.num_epochs = num_epochs

        # Training follow-up
        self.epoch = 1  # starts at 1
        self.batch_index = 1  # starts at 1
        self.best_model_epoch = None
        self.training_time = 0
        self.score = None
        self.additional_score = 0

        # Get git hash
        try:
            import git

            current_file_path = Path(__file__).parent.resolve()
            repo = git.Repo(current_file_path, search_parent_directories=True)
            self.git_hash = repo.head.object.hexsha
            print(f"Current commit hash: {self.git_hash}")
        except:  # if not a git repository
            self.git_hash = "unknown"

    def check_validity(self) -> None:
        if self.num_batches_train is None:
            raise ValueError("Number of batches is not defined yet")

    def get_total_batches(self) -> int:
        self.check_validity()
        return self.num_batches_train * self.num_epochs

    def get_current_batch(self) -> int:
        self.check_validity()
        return (self.epoch - 1) * self.num_batches_train + self.batch_index
