import json
import os

def load_material(name):
    data_path = os.path.join(os.path.dirname(__file__), "data", "plastics.json")
    with open(data_path, 'r') as f:
        plastics = json.load(f)
    if name not in plastics:
        raise ValueError("Unknown plastic type")
    return plastics[name]
