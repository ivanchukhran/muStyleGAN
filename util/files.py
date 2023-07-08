import json
import os


def fetch_json(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            file = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
    return file


def dump_json(path: str, data: dict) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error writing file: {e}")


def read_settings(path: str) -> dict:
    """
    Read the settings from the given path.
    :param path: The path to the settings file.
    :return: settings: dict - The settings.
    """
    try:
        with open(path, 'r') as f:
            settings = json.load(f)
    except Exception as e:
        print(f"Error reading settings file: {e}")
        settings = {}
    return settings


def create_dir_or_ignore(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"Directory {path} already exists.")


def filter_by_dirname(path, dir_name) -> list[str]:
    return [folder for folder in os.listdir(path) if dir_name in folder]
