import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Load your dataset
file_path = "./datasets/cleaned_raw_data.csv"
data = pd.read_csv(file_path)

# Assuming "quality" is the target variable and other columns are features
X = data.drop(columns=["quality"])
y = data["quality"]

# Calculate information gain (mutual information) for each feature
info_gain = mutual_info_classif(X, y)

# Create a DataFrame to store feature names and their information gain
info_gain_df = pd.DataFrame({"Feature": X.columns, "Information Gain": info_gain})

# Sort the DataFrame by information gain in descending order
info_gain_df = info_gain_df.sort_values(by="Information Gain", ascending=False)

# Print information gain for each feature
for index, row in info_gain_df.iterrows():
    print(f"Information Gain for '{row['Feature']}': {row['Information Gain']:.4f}")
