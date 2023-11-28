import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Load the trained KMeans model
model_file_path = "classification/kmeans_classification/kmeans_model.pkl"
kmeans_model = joblib.load(model_file_path)

# Read the test data
file_path = "classification/kmeans_classification/test_data.csv"
test_data = pd.read_csv(file_path)

# Assuming 'quality' is the target variable and other columns are features
X_test = test_data.drop(columns=["quality"])

# Normalize features using the same scaler used for training
scaler = StandardScaler()
X_test_normalized = scaler.fit_transform(X_test)

# Get the cluster labels for each data point
cluster_labels = kmeans_model.predict(X_test_normalized)

# Add the cluster labels to the test dataset
test_data_with_clusters = test_data.copy()
test_data_with_clusters["Cluster"] = cluster_labels

# Assuming 'quality' is the target variable in the test data
y_test = test_data["quality"]

start_time = time.time()
# Evaluate accuracy based on the majority quality in each cluster
cluster_majority_quality = test_data_with_clusters.groupby("Cluster")["quality"].apply(
    lambda x: x.mode().iloc[0] if not x.mode().empty else None
)
y_pred = cluster_majority_quality.loc[cluster_labels].values
end_time = time.time()
print("Time taken to predict:", end_time - start_time)

# Remove rows with missing predictions
valid_indices = ~pd.isna(y_pred)
y_test = y_test[valid_indices]
y_pred = y_pred[valid_indices]

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2%}")

# Print accuracy for each value of the target attribute
unique_values = test_data["quality"].unique()
for value in unique_values:
    indices = test_data["quality"] == value
    accuracy_value = accuracy_score(test_data.loc[indices, "quality"], y_pred[indices])
    print(f"Accuracy for '{value}': {accuracy_value:.2%}")
