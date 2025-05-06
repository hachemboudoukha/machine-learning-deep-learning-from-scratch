import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, mean_squared_error

class KNN:
    def __init__(self, k=5, task='classification'):
        """
        Initialize KNN classifier/regressor
        
        Parameters:
        -----------
        k : int
            Number of neighbors to consider
        task : str
            'classification' or 'regression'
        """
        self.k = k
        self.task = task
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        Store training data
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _get_neighbors(self, x):
        """Get k nearest neighbors for a given point"""
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
    
    def predict(self, X):
        """
        Make predictions for new data
        
        Parameters:
        -----------
        X : array-like
            Features to predict
        
        Returns:
        --------
        predictions : array
            Predicted labels/values
        """
        X = np.array(X)
        predictions = []
        
        for x in X:
            neighbors = self._get_neighbors(x)
            
            if self.task == 'classification':
                # Get most common class among neighbors
                neighbor_labels = [n[1] for n in neighbors]
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                predictions.append(most_common)
            else:  # regression
                # Calculate mean of neighbor values
                neighbor_values = [n[1] for n in neighbors]
                predictions.append(np.mean(neighbor_values))
                
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Calculate accuracy (classification) or MSE (regression)
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            True labels/values
        
        Returns:
        --------
        score : float
            Accuracy or negative MSE
        """
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            return accuracy_score(y, y_pred)
        else:
            return -mean_squared_error(y, y_pred)  # Negative MSE for consistency with sklearn