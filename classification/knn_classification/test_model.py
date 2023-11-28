import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Load the trained k-NN model
model_file_path = "classification/knn_classification/knn_model.pkl"
knn_model = joblib.load(model_file_path)

# Read the test data
test_file_path = "classification/knn_classification/test_data.csv"
test_data = pd.read_csv(test_file_path)

# Assuming 'quality' is the target variable and other columns are features
X_test = test_data.drop(columns=["quality"])

# Normalize features using the same scaler used for training
scaler = StandardScaler()
X_test_normalized = scaler.fit_transform(X_test)

# Make predictions on the test set with normalized features
start_time = time.time()
y_test_pred = knn_model.predict(X_test_normalized)
end_time = time.time()
print("Time taken to predict:", end_time - start_time)

# Evaluate on the test set
accuracy_test = accuracy_score(test_data["quality"], y_test_pred)
print(f"Accuracy on the test set: {accuracy_test:.2%}")

# Print accuracy for each value of the target attribute
unique_values = test_data["quality"].unique()
for value in unique_values:
    indices = test_data["quality"] == value
    accuracy_value = accuracy_score(
        test_data.loc[indices, "quality"], y_test_pred[indices]
    )
    print(f"Accuracy for '{value}': {accuracy_value:.2%}")


"""'
Information Gain for 'alcohol': 0.1896
Information Gain for 'volatile acidity': 0.1124
Information Gain for 'citric acid': 0.0884
Information Gain for 'total sulfur dioxide': 0.0881
Information Gain for 'sulphates': 0.0856
Information Gain for 'density': 0.0654
Information Gain for 'free sulfur dioxide': 0.0513
Information Gain for 'chlorides': 0.0443
Information Gain for 'fixed acidity': 0.0348
Information Gain for 'pH': 0.0099
Information Gain for 'residual sugar': 0.0034
"""
