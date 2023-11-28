import pandas as pd
import matplotlib.pyplot as plt

# Load the wine quality dataset
file_path = "datasets/cleaned_raw_data.csv"
data = pd.read_csv(file_path)

# Assuming 'quality' is the target variable
quality_counts = data["quality"].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(quality_counts, labels=quality_counts.index, startangle=140)
plt.title("Distribution of Wine Qualities")
plt.show()
