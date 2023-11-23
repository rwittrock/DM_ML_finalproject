import pandas as pd
import numpy as np


def replace_outliers_with_mean(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Replace outliers with the mean of the column
    data[column] = np.where(
        (data[column] < lower_bound) | (data[column] > upper_bound),
        data[column].mean(),
        data[column],
    )
    return data


# Read the CSV file
file_path = "./datasets/raw_data.csv"
data = pd.read_csv(file_path)

# Remove entries with missing values
data = data.dropna()

# Round down all numbers in the "chlorides" column to 3 decimals
data["chlorides"] = data["chlorides"].round(3)

# Replace outliers with the mean using the IQR method for specified columns
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
    data = replace_outliers_with_mean(data, column)

# Save the cleaned data to a new CSV file
cleaned_file_path = "./datasets/cleaned_raw_data.csv"
data.to_csv(cleaned_file_path, index=False)

print("Cleaning completed. Cleaned data saved to", cleaned_file_path)
