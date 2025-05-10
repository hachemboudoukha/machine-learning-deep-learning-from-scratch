import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from k_means_clustering import KMeans

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Initialize and train the model
kmeans = KMeans(n_clusters=4, max_iters=100, init='kmeans++', random_state=42)
kmeans.fit(X)

# Get predictions
predictions = kmeans.predict(X)

# Calculate metrics
silhouette_avg = silhouette_score(X, predictions)
print(f"Silhouette Score: {silhouette_avg}")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show() 