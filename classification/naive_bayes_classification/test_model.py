import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained Naive Bayes model
model_file_path = "classification/naive_bayes_classification/naive_bayes_model.pkl"
naive_bayes_model = joblib.load(model_file_path)

# Read the test data
test_file_path = "classification/naive_bayes_classification/test_data.csv"
test_data = pd.read_csv(test_file_path)

# Assuming 'quality' is the target variable and other columns are features
X_test = test_data.drop(columns=["quality"])

# Normalize features using the same scaler used for training (even though Naive Bayes is not sensitive to scaling)
scaler = StandardScaler()
X_test_normalized = scaler.fit_transform(X_test)

# Make predictions on the test set with normalized features
y_test_pred = naive_bayes_model.predict(X_test_normalized)

# Evaluate on the test set
accuracy_test = accuracy_score(test_data["quality"], y_test_pred)
print(f"Overall Accuracy on the test set: {accuracy_test:.2%}")

# Print accuracy for each value of the target attribute
unique_values = test_data["quality"].unique()
for value in unique_values:
    indices = test_data["quality"] == value
    accuracy_value = accuracy_score(
        test_data.loc[indices, "quality"], y_test_pred[indices]
    )
    print(f"Accuracy for '{value}': {accuracy_value:.2%}")
