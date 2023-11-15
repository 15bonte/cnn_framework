import os

from ..utils.create_dummy_data_set import generate_data_set

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_folder_path(folder_name: str) -> None:
    """
    Returns absolute path to folder file.
    """
    local_path = os.path.join(CURRENT_DIR, folder_name)

    if not os.path.exists(local_path) and folder_name == "images":
        generate_data_set(local_path)

    os.makedirs(local_path, exist_ok=True)

    return local_path
