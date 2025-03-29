import yaml
from typing import Tuple


def find_keys_path() -> str:
    return "keys.yaml"

def load_deepseek_api_key() -> str:
    with open(find_keys_path(), "r") as f:
        keys = yaml.safe_load(f)
    return keys["DEEPSEEK_API_KEY"]

def load_google_cse_keys() -> Tuple[str, str]:
    """
    Load the Google API key and Custom Search Engine ID from keys.yaml
    returns: GOOGLE_API_KEY, GOOGLE_CSE_ID
    """
    with open(find_keys_path(), "r") as f:
        keys = yaml.safe_load(f)
        if "GOOGLE_API_KEY" in keys and "GOOGLE_CSE_ID" in keys:
            GOOGLE_API_KEY = keys["GOOGLE_API_KEY"]
            GOOGLE_CSE_ID = keys["GOOGLE_CSE_ID"]
        else:
            raise ValueError("GOOGLE_API_KEY and GOOGLE_CSE_ID must be defined in keys.yaml")
    return GOOGLE_API_KEY, GOOGLE_CSE_ID