from joblib import dump, load
import json
from .paths import get_config_path


def save_obj(obj, filepath):
    """
    Save model locally from to joblib object
    
    :param filepath:

    """
    obj  = dump(obj, filepath)


def load_obj(filepath):
    """
    Load model locally from a joblib object
    
    :param filepath:

    """
    obj = load(filepath)
    return obj


def save_json(json_file, filepath):
    """
    Save json file as json

    :param json_file:
    :param filepath:

    """
    with open(filepath, "w", encoding='utf-8') as f:
        json.dump(json_file, f, indent=4)


def find_default(parameter_name):
    """
    Get default value for a given parameter

    :param parameter_name:

    :return: Default value
    """

    config_path = get_config_path()
    with open(config_path, encoding='utf-8') as f:
        conf = json.load(f)
    return conf[parameter_name]
    