import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, init='kmeans++', random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def _initialize_centroids(self, X):
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            np.random.seed(self.random_state)
            idx = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[idx]
            
        elif self.init == 'kmeans++':
            np.random.seed(self.random_state)
            self.centroids = [X[np.random.randint(n_samples)]]
            
            for _ in range(self.n_clusters - 1):
                distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in self.centroids]) for x in X])
                probs = distances / distances.sum()
                cumulative_probs = np.cumsum(probs)
                r = np.random.random()
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        self.centroids.append(X[j])
                        break
    
    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1) ** 2
        return distances
    
    def _assign_clusters(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                new_centroids[k] = np.mean(X[labels == k], axis=0)
        return new_centroids
    
    def fit(self, X):
        self._initialize_centroids(X)
        self.labels = np.zeros(X.shape[0])
        
        for _ in range(self.max_iters):
            old_labels = self.labels.copy()
            self.labels = self._assign_clusters(X)
            
            if np.all(old_labels == self.labels):
                break
            
            self.centroids = self._update_centroids(X, self.labels)
    
    def predict(self, X):
        return self._assign_clusters(X) 