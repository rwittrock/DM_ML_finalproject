import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Read the training data
file_path = "classification/neural_network_classification/training_data.csv"
data = pd.read_csv(file_path)

# Assuming 'quality' is the target variable and other columns are features
X = data.drop(columns=["quality"])
y = data["quality"]

# Convert 'quality' to categorical
y = pd.Categorical(y)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_valid_normalized = scaler.transform(X_valid)

# Build a neural network model for classification
model = Sequential()
# model.add(Dense(128, input_dim=X_train_normalized.shape[1], activation="relu"))
model.add(Dense(64, input_dim=X_train_normalized.shape[1], activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(
    Dense(len(y.unique()), activation="softmax")
)  # Softmax activation for classification

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Convert labels to one-hot encoding
y_train_encoded = pd.get_dummies(y_train)
y_valid_encoded = pd.get_dummies(y_valid)

# Train the model
model.fit(
    X_train_normalized,
    y_train_encoded,
    epochs=50,
    batch_size=32,
    validation_data=(X_valid_normalized, y_valid_encoded),
)

# Evaluate the model on the validation set
accuracy = model.evaluate(X_valid_normalized, y_valid_encoded, verbose=0)[1]
print(f"Accuracy on the validation set: {accuracy:.2%}")

# Save the trained model to a file
model.save(
    "classification/neural_network_classification/wine_quality_prediction_model_classification.h5"
)
print("Neural network classification model created and saved.")
