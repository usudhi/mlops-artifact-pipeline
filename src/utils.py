import json
import pickle
from sklearn.datasets import load_digits

def load_config(path="config/config.json"):
    """Load JSON config file."""
    with open(path, "r") as f:
        return json.load(f)

def load_data():
    """Load digits dataset (features and labels)."""
    digits = load_digits()
    return digits.data, digits.target

def save_model(model, path="model_train.pkl"):
    """Save trained model to disk."""
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path="model_train.pkl"):
    """Load trained model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
