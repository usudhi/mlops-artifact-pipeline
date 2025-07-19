from utils import load_model, load_data

# Load model and data
model = load_model()
X, _ = load_data()

# Predict
predictions = model.predict(X)

print(" Sample Predictions (first 10):", predictions[:10])
