from pathlib import Path


def take_all_files_paths(path_folder: str) -> list:
    folder = Path(path_folder)
    all_file = [str(f) for f in folder.rglob("*") if f.is_file()]
    return all_file
