import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Load the trained k-NN model
model_file_path = "classification/knn_classification/knn_model.pkl"
knn_model = joblib.load(model_file_path)

# Read the test data
test_file_path = "classification/test_data.csv"
test_data = pd.read_csv(test_file_path)

# Assuming 'quality' is the target variable and other columns are features
X_test = test_data.drop(columns=["quality"])
y_test = test_data["quality"].astype(int)  # Convert to integer if needed

# Make predictions on the test set
y_test_pred = knn_model.predict(X_test)

# Evaluate on the test set
accuracy_test = accuracy_score(y_test, y_test_pred)

print(f"Accuracy on the test set: {accuracy_test:.2%}")
