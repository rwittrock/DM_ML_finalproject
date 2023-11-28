import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the wine quality dataset
file_path = "datasets/cleaned_raw_data.csv"
data = pd.read_csv(file_path)

# Assuming 'quality' and 'alcohol' are the relevant columns
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x="quality", y="alcohol", data=data, palette="Set3")
plt.title("Boxplot of Alcohol Content by Wine Quality")
plt.xlabel("Wine Quality")
plt.ylabel("Alcohol Content")
plt.show()
