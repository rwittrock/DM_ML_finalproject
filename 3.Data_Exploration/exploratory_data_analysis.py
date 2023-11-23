#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
file_path = "WineQT.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print("First few rows of the dataset:")
print(df.head())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Pairplot for visualizing relationships between variables
sns.pairplot(df, hue='quality', markers='o')
plt.title("Pairplot of Variables")
plt.show()

# Correlation heatmap to visualize the correlation between variables
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Boxplot for quality distribution
plt.figure(figsize=(8, 6))
sns.boxplot(x='quality', y='alcohol', data=df)
plt.title("Boxplot of Alcohol Content by Quality")
plt.show()
