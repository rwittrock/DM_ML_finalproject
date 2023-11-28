import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import time

# Read the training data
file_path = "classification/knn_classification/training_data.csv"
data = pd.read_csv(file_path)

# Assuming 'quality' is the target variable and other columns are features
X = data.drop(columns=["quality"])
y = data["quality"]

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_valid_normalized = scaler.transform(X_valid)


# Hyperparameter tuning for k-NN
parameters = {"n_neighbors": [49]}
knn_model = KNeighborsClassifier()
grid_search = GridSearchCV(knn_model, parameters, cv=5)
grid_search.fit(X_train_normalized, y_train)

# Print the best hyperparameters
print("Best n_neighbors:", grid_search.best_params_)

start_time = time.time()
# Train the model with the best hyperparameters
best_knn_model = grid_search.best_estimator_
best_knn_model.fit(X_train_normalized, y_train)
end_time = time.time()
print("Time taken to train:", end_time - start_time)

# Make predictions on the validation set with normalized features
y_pred = best_knn_model.predict(X_valid_normalized)

# Save the trained model to a file
model_file_path = "classification/knn_classification/knn_model.pkl"
joblib.dump(best_knn_model, model_file_path)

print("Tuned k-NN model created and saved to", model_file_path)
