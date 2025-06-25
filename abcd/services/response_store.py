import json
import os

def save_response(path: str, response: dict):
    """
    Saves the response JSON to the given path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(response, f, indent=2)
