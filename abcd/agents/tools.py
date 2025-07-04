from langchain.tools import tool
import os
import pickle

@tool
def load_ocr_data(pickle_path: str) -> dict:
    """Loads OCR data from a pickle file."""
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    return {}

@tool
def search_text(ocr_data: dict, query: str) -> str:
    """Searches the OCR text for the given query."""
    matches = []
    for page, content in ocr_data.items():
        if query.lower() in content.lower():
            matches.append(f"[{page}]\n{content}")
    return "\n\n".join(matches) if matches else "No match found."

@tool
def save_json_response(path: str, response: dict) -> str:
    """Saves a response dict as JSON at the given path."""
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(response, f, indent=2)
    return f"Response saved to {path}"
