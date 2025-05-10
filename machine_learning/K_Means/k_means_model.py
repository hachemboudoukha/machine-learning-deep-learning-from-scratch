import numpy as np
from sklearn.metrics import silhouette_score

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None
        
    def initialize_centroids(self, X):
        # Randomly select k points as initial centroids
        n_samples = X.shape[0]
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        return X[random_indices]
    
    def compute_distance(self, X, centroids):
        # Compute distances between each point and each centroid
        distances = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            distances[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)
        return distances
    
    def compute_centroids(self, X, labels):
        # Compute new centroids as mean of points in each cluster
        centroids = np.zeros((self.k, X.shape[1]))
        for k in range(self.k):
            if np.sum(labels == k) > 0:
                centroids[k] = np.mean(X[labels == k], axis=0)
        return centroids
    
    def fit(self, X):
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = self.compute_distance(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = self.compute_centroids(X, self.labels)
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        distances = self.compute_distance(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def score(self, X):
        """
        Compute the silhouette score for the clustering
        """
        if self.labels is None:
            self.fit(X)
        return silhouette_score(X, self.labels) 