#5
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Generate random data and labels
np.random.seed(42)
x_all = np.random.rand(100).reshape(-1, 1)
x_train, x_test = x_all[:50], x_all[50:]
y_train = np.where(x_train <= 0.5, 1, 2)


# Try different k values
k_values = [1, 2, 3, 4, 5, 20, 30]
results = {}

print("k-NN Classification Results:\n")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results[k] = y_pred
    print(f"k = {k}:\nPredicted Classes: {y_pred}\n" + "-" * 50)

# Visualize for k = 5
plt.figure(figsize=(10, 4))
plt.title("KNN Classification (k=5)")
plt.scatter(x_train, y_train, color='blue', label='Train Data')
plt.scatter(x_test, results[5], color='red', marker='x', label='Test Predictions')
plt.xlabel('x value')
plt.ylabel('Class')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
