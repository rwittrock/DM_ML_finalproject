import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Read the training data
file_path = "classification/training_data.csv"
data = pd.read_csv(file_path)

# Assuming 'quality' is the target variable and other columns are features
X = data.drop(columns=["quality"])
y = data["quality"]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Determine the value of k as the square root of the number of training points (rounded to the nearest odd integer)
k_value = int(np.sqrt(len(X_train)))
k_value = k_value + 1 if k_value % 2 == 0 else k_value  # Ensure k is odd

# Create the k-NN model
knn_model = KNeighborsClassifier(n_neighbors=k_value)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = knn_model.predict(X_valid)

# Calculate accuracy on the validation set (replace with an appropriate metric for classification)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Accuracy on the validation set: {accuracy:.2%}")

# Save the trained model to a file
model_file_path = "classification/knn_classification/knn_model.pkl"
joblib.dump(knn_model, model_file_path)

print("k-NN model created and saved to", model_file_path)
