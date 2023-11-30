import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# import matplotlib.pyplot as plt

# # Read the CSV file into a DataFrame
# file_path = "../datasets/cleaned_raw_data.csv"  # Replace with the actual file path
# df = pd.read_csv(file_path)

# # Assuming 'quality' is the target variable
# # Create a binary classification target variable: 0 for low quality (<=5), 1 for high quality (>5)
# df['quality_class'] = df['quality'].apply(lambda x: 1 if x > 5 else 0)

# # Features and target variable
# X = df.drop(['quality', 'quality_class', 'Id'], axis=1)
# y = df['quality_class']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a Gradient Boosting Classifier
# gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# # Train the model
# gb_classifier.fit(X_train, y_train)

# # Get feature importances
# feature_importances = gb_classifier.feature_importances_

# # Create a DataFrame to display feature importances
# feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
# feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # Plot feature importances
# plt.figure(figsize=(10, 6))
# plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
# plt.xlabel('Importance')
# plt.title('Feature Importances (Gradient Boosting)')
# plt.show()
