import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_folder_path(folder_name: str) -> None:
    """
    Returns absolute path to folder file.
    """
    local_path = os.path.join(CURRENT_DIR, folder_name)

    os.makedirs(local_path, exist_ok=True)

    return local_path
