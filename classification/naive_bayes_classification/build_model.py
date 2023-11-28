import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib as joblib

# Read the training data
file_path = "classification/naive_bayes_classification/training_data.csv"
data = pd.read_csv(file_path)

# quality is target attribute
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

# Create the Naive Bayes model
naive_bayes_model = GaussianNB()

# Train the model on normalized features
naive_bayes_model.fit(X_train_normalized, y_train)

# Make predictions on the validation set with normalized features
y_pred = naive_bayes_model.predict(X_valid_normalized)

# Evaluate the model
accuracy = accuracy_score(y_valid, y_pred)
print(f"Accuracy on the validation set: {accuracy:.2%}")

model_file_path = "classification/naive_bayes_classification/naive_bayes_model.pkl"
joblib.dump(naive_bayes_model, model_file_path)

print("Naive Bayes model created and saved to", model_file_path)
