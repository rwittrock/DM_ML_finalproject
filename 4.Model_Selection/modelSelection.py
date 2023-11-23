import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the CSV file into a DataFrame
file_path = "../datasets/cleaned_raw_data.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# Assuming 'quality' is the target variable
# Create a binary classification target variable: 0 for low quality (<=5), 1 for high quality (>5)
df['quality_class'] = df['quality'].apply(lambda x: 1 if x > 5 else 0)

# Features and target variable
X = df.drop(['quality', 'quality_class', 'Id'], axis=1)
y = df['quality_class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
