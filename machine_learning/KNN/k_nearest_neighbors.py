import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5, task='classification'):
        self.k = k
        self.task = task
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _get_neighbors(self, x):
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x in X:
            neighbors = self._get_neighbors(x)
            
            if self.task == 'classification':
                neighbor_labels = [n[1] for n in neighbors]
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                predictions.append(most_common)
            else:  # regression
                neighbor_values = [n[1] for n in neighbors]
                predictions.append(np.mean(neighbor_values))
                
        return np.array(predictions) 