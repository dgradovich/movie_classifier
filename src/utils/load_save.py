from joblib import dump, load
import json

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
