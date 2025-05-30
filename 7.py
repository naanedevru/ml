#7
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# === Linear Regression: California Housing ===
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data[:, data.feature_names.index('AveRooms')].reshape(-1, 1),
    data.target, test_size=0.2, random_state=42
)

lr_model = LinearRegression().fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

print("=== Linear Regression on California Housing ===")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Average Rooms')
plt.ylabel('House Value')
plt.title('Linear Regression - California Housing')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Polynomial Regression: Auto MPG ===
auto = sns.load_dataset('mpg').dropna(subset=['horsepower', 'mpg'])
auto['horsepower'] = pd.to_numeric(auto['horsepower'], errors='coerce')
auto = auto.dropna(subset=['horsepower'])

X = auto[['horsepower']].values
y = auto['mpg'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression().fit(X_poly_train, y_train)
y_poly_pred = poly_model.predict(X_poly_test)

print("\n=== Polynomial Regression on Auto MPG ===")
print("MSE:", mean_squared_error(y_test, y_poly_pred))
print("R²:", r2_score(y_test, y_poly_pred))

sorted_idx = X_test[:, 0].argsort()
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual MPG')
plt.plot(X_test[sorted_idx], y_poly_pred[sorted_idx], color='green', linewidth=2, label='Predicted MPG')
plt.xlabel('Horsepower')
plt.ylabel('Miles per Gallon (MPG)')
plt.title('Polynomial Regression - Auto MPG Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
