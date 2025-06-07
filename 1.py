#1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
data = fetch_california_housing(as_frame=True).data

# Create histograms for all numerical features
for column in data.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(data[column], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

# Generate box plots for all numerical features
for column in data.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data[column], color='skyblue')
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.grid()
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
