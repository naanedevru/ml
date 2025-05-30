#3
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load and standardize data
iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

# Apply PCA
X_pca = PCA(n_components=2).fit_transform(X)

# Plot PCA results
plt.figure(figsize=(8, 6))
for i, color in enumerate(['red', 'green', 'blue']):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                color=color, label=iris.target_names[i])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset (4D → 2D)')
plt.legend()
plt.show()

# Explained variance
print("Explained variance ratio:", PCA(n_components=2).fit(StandardScaler().fit_transform(iris.data)).explained_variance_ratio_)
