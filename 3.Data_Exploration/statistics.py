# %%

import pandas as pd

# Read the CSV file into a DataFrame
file_path = "datasets/cleaned_raw_data.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print("First few rows of the dataset:")
print(df.head())

# Basic statistics
print("\nBasic statistics:")
print(df.describe().round(2))

# Median
print("\nMedian:")
print(df.median().round(2))

# Mode
print("\nMode:")
print(df.mode().iloc[0].round(2))

# Quartiles
print("\nQuartiles:")
print(df.quantile([0.25, 0.5, 0.75]).round(2))

# Display any other specific statistics as needed
# For example, mean
print("\nMean:")
print(df.mean().round(2))
