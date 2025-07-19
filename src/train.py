from sklearn.linear_model import LogisticRegression
from utils import load_config, load_data, save_model
from sklearn.metrics import accuracy_score, f1_score

# Load config and data
config = load_config()
X, y = load_data()

# Split for evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(
    C=config["C"],
    solver=config["solver"],
    max_iter=config["max_iter"]
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

# Print metrics
print(f" Model trained successfully")
print(f" Accuracy: {accuracy:.4f}")
print(f" F1 Score: {f1:.4f}")

# Save model
save_model(model)


