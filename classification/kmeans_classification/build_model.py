import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Load the wine quality dataset
file_path = "classification/kmeans_classification/training_data.csv"
data = pd.read_csv(file_path)

# Assuming 'quality' is the target variable and other columns are features
X = data.drop(columns=["quality"])

# Normalize features using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Choose the number of clusters (k)
k = 6  # Replace with the desired number of clusters

# Create the k-means model
kmeans_model = KMeans(n_clusters=k, random_state=42)

# Fit the model to the normalized data
kmeans_model.fit(X_normalized)

# Get the cluster labels for each data point
cluster_labels = kmeans_model.labels_

# Add the cluster labels to the original dataset
data_with_clusters = data.copy()
data_with_clusters["Cluster"] = cluster_labels

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_normalized, data_with_clusters["quality"], test_size=0.2, random_state=42
)

# Create a random forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

start_time = time.time()
# Train the classifier
rf_classifier.fit(X_train, y_train)
end_time = time.time()
print("Time taken to train:", end_time - start_time)

# Make predictions on the validation set
y_pred = rf_classifier.predict(X_valid)

# Calculate accuracy
accuracy = accuracy_score(y_valid, y_pred)
print(f"Accuracy on the validation set: {accuracy:.2%}")

# Save the trained model to a file
model_file_path = "classification/kmeans_classification/kmeans_model.pkl"
joblib.dump(rf_classifier, model_file_path)

print("Random Forest classifier created and saved to", model_file_path)
