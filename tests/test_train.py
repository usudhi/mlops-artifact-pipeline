import json
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits

def load_config():
    with open("config/config.json", "r") as f:
        return json.load(f)

def test_config_file_loads():
    config = load_config()
    assert isinstance(config["C"], float), "C should be a float"
    assert isinstance(config["solver"], str), "solver should be a string"
    assert isinstance(config["max_iter"], int), "max_iter should be an int"

def test_model_training_and_accuracy():
    config = load_config()
    digits = load_digits()
    X, y = digits.data, digits.target

    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    accuracy = model.score(X, y)

    assert isinstance(model, LogisticRegression), "Model should be LogisticRegression"
    assert accuracy > 0.9, f"Accuracy too low: {accuracy}"