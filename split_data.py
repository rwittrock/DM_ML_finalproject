import pandas as pd
import numpy as np

# Read the CSV file
file_path = "./datasets/cleaned_raw_data.csv"
data = pd.read_csv(file_path)
# remove attributes from data
data = data.drop(
    columns=[
        "residual sugar",
        "pH",
        "fixed acidity",
        "chlorides",
        "free sulfur dioxide",
        "density",
        "sulphates",
    ]
)

# Randomly shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate the index to split the data into training and test sets
split_index = int(2 / 3 * len(data))

# Split the data
training_data = data.iloc[:split_index, :]
test_data = data.iloc[split_index:, :]

# Save the training and test data to CSV files
training_file_path = "./classification/training_data.csv"
test_file_path = "./classification/test_data.csv"

training_data.to_csv(training_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print(
    "Data split into training and test sets. Training data saved to", training_file_path
)
print("Test data saved to", test_file_path)
