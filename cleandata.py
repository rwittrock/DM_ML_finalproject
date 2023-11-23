import pandas as pd

# Read the CSV file
file_path = "./datasets/raw_data.csv"
data = pd.read_csv(file_path)

# Round down all numbers in the "chlorides" column to 3 decimals
data["chlorides"] = data["chlorides"].round(3)

# Replace missing values with the mean of each column
for column in data.columns:
    if data[column].isnull().any():
        mean_value = data[column].mean()
        data[column].fillna(mean_value, inplace=True)
        print(f"Replaced missing values in '{column}' with the mean ({mean_value:.3f})")

# Save the cleaned data to a new CSV file
cleaned_file_path = "./datasets/cleaned_raw_data.csv"
data.to_csv(cleaned_file_path, index=False)

print("Cleaning completed. Cleaned data saved to", cleaned_file_path)
