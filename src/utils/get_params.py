import json

def get_params_dict(path):
    """
    Loads a json file with parameters for pretraining.    
    """
    try:
        with open(path, "r") as f:
            params = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("No params file found at {}".format(path))

    return params
