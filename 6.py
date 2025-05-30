#6
import numpy as np
import matplotlib.pyplot as plt

def kernel(x, x_i, tau):
    return np.exp(-((x - x_i) ** 2) / (2 * tau ** 2))

def locally_weighted_regression(x_train, y_train, x_query, tau):
    X = np.c_[np.ones(len(x_train)), x_train]
    x_q = np.array([1, x_query])
    W = np.diag(kernel(x_query, x_train, tau))
    try:
        theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y_train
        return x_q @ theta
    except np.linalg.LinAlgError:
        return 0

# Generate data
np.random.seed(1)
x = np.linspace(-3, 3, 100)
y = np.sin(x) + 0.1 * np.random.randn(100)
x_query = np.linspace(-3, 3, 200)

# Run LWR
tau = 0.5
y_pred = [locally_weighted_regression(x, y, xq, tau) for xq in x_query]

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='lightgray', label='Noisy Training Data')
plt.plot(x, np.sin(x), 'g--', label='True sin(x)')
plt.plot(x_query, y_pred, 'r', label='LWR Prediction')
plt.title(f'Locally Weighted Regression (tau = {tau:.2f})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
