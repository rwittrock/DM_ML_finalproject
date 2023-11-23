import pandas as pd
import numpy as np


def remove_outliers_iqr(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data[column] = np.where(
        (data[column] < lower_bound) | (data[column] > upper_bound),
        np.nan,
        data[column],
    )
    return data


# Read the CSV file
file_path = "./datasets/raw_data.csv"
data = pd.read_csv(file_path)

# Round down all numbers in the "chlorides" column to 3 decimals
data["chlorides"] = data["chlorides"].round(3)

# Remove outliers using the IQR method for specified columns
columns_to_process = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]

for column in columns_to_process:
    data = remove_outliers_iqr(data, column)

# Replace missing values with the mean of each column
data = data.fillna(data.mean())

# Save the cleaned data to a new CSV file
cleaned_file_path = "./datasets/cleaned_raw_data.csv"
data.to_csv(cleaned_file_path, index=False)

print("Cleaning completed. Cleaned data saved to", cleaned_file_path)
