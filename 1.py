#1
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
data = fetch_california_housing(as_frame=True).data

# Plot histograms and box plots for all features
for column in data.columns:
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axs[0].hist(data[column], bins=30, color='skyblue', edgecolor='black')
    axs[0].set_title(f'Distribution of {column}')
    axs[0].set_xlabel(column)
    axs[0].set_ylabel('Frequency')
    axs[0].grid()

    # Box Plot
    sns.boxplot(x=data[column], ax=axs[1], color='skyblue')
    axs[1].set_title(f'Box Plot of {column}')
    axs[1].set_xlabel(column)
    axs[1].grid()

    plt.tight_layout()
    plt.show()

# Analyze and print potential outliers
print("Potential outliers:")
for column in data.columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    if not outliers.empty:
        print(f"{column}: {len(outliers)} outliers detected")
