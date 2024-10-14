from .default_data_manager import DefaultDataManager


class ClassificationDataManager(DefaultDataManager):
    """This class is used to manage data for classification.
    Files not terminated by "_c*" are ignored."""

    def get_distinct_files(self):
        files = super().get_distinct_files()
        return [f for f in files if "_c" in f]
