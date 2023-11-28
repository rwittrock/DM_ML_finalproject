import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import time

# Load the trained neural network classification model
model_file_path = "classification/neural_network_classification/wine_quality_prediction_model_classification.h5"
model = load_model(model_file_path)

# Read the test data
test_file_path = "classification/neural_network_classification/test_data.csv"
test_data = pd.read_csv(test_file_path)

# Assuming 'quality' is the target variable and other columns are features
X_test = test_data.drop(columns=["quality"])
y_test = test_data["quality"]

# Normalize features using the same scaler used for training
scaler = StandardScaler()
X_test_normalized = scaler.fit_transform(X_test)

# Convert 'quality' to categorical
y_test_categorical = pd.Categorical(y_test)
y_test_encoded = pd.get_dummies(y_test_categorical)

start_time = time.time()
# Make predictions on the test set with normalized features
y_pred_proba = model.predict(X_test_normalized)
y_pred = y_pred_proba.argmax(axis=1)  # Convert probabilities to class labels
end_time = time.time()
print("Time taken to predict:", end_time - start_time)

# Convert predictions to categorical for comparison
y_pred_categorical = pd.Categorical.from_codes(
    y_pred, categories=y_test_categorical.categories
)

# Evaluate on the test set
accuracy = accuracy_score(y_test_categorical, y_pred_categorical)
print(f"Accuracy on the test set: {accuracy:.2%}")

# Print accuracy for each value of the target attribute
unique_values = test_data["quality"].unique()
for value in unique_values:
    indices = test_data["quality"] == value
    accuracy_value = accuracy_score(
        test_data.loc[indices, "quality"], y_pred_categorical[indices]
    )
    print(f"Accuracy for '{value}': {accuracy_value:.2%}")
