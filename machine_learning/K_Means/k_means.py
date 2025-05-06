import numpy as np
from sklearn.metrics import silhouette_score

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, init='kmeans++', random_state=None):
        """
        Initialize K-Means clustering
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        max_iters : int
            Maximum number of iterations
        init : str
            Initialization method ('kmeans++' or 'random')
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def _initialize_centroids(self, X):
        """Initialize centroids using specified method"""
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Random initialization
            np.random.seed(self.random_state)
            idx = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[idx]
            
        elif self.init == 'kmeans++':
            # K-means++ initialization
            np.random.seed(self.random_state)
            self.centroids = [X[np.random.randint(n_samples)]]
            
            for _ in range(self.n_clusters - 1):
                # Calculate distances to nearest centroid
                distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in self.centroids]) for x in X])
                # Calculate probabilities
                probs = distances / distances.sum()
                # Choose new centroid
                cumulative_probs = np.cumsum(probs)
                r = np.random.random()
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        self.centroids.append(X[j])
                        break
    
    def _compute_distances(self, X):
        """Compute distances between points and centroids"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1) ** 2
        return distances
    
    def _assign_clusters(self, X):
        """Assign points to nearest centroid"""
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """Update centroids based on mean of assigned points"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                new_centroids[k] = np.mean(X[labels == k], axis=0)
        return new_centroids
    
    def fit(self, X):
        """
        Fit K-Means clustering to the data
        
        Parameters:
        -----------
        X : array-like
            Training data
        """
        # Initialize centroids
        self._initialize_centroids(X)
        
        # Initialize labels
        self.labels = np.zeros(X.shape[0])
        
        # Main loop
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            old_labels = self.labels.copy()
            self.labels = self._assign_clusters(X)
            
            # Check for convergence
            if np.all(old_labels == self.labels):
                break
            
            # Update centroids
            self.centroids = self._update_centroids(X, self.labels)
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Parameters:
        -----------
        X : array-like
            Data to predict
        
        Returns:
        --------
        labels : array
            Predicted cluster labels
        """
        return self._assign_clusters(X)
    
    def score(self, X):
        """
        Calculate silhouette score
        
        Parameters:
        -----------
        X : array-like
            Data to score
        
        Returns:
        --------
        score : float
            Silhouette score
        """
        labels = self.predict(X)
        return silhouette_score(X, labels)
    
    def inertia(self, X):
        """
        Calculate inertia (within-cluster sum of squares)
        
        Parameters:
        -----------
        X : array-like
            Data to calculate inertia for
        
        Returns:
        --------
        inertia : float
            Within-cluster sum of squares
        """
        distances = self._compute_distances(X)
        return np.sum(np.min(distances, axis=1)) 