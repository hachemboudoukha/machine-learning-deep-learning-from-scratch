import numpy as np
from sklearn.metrics import accuracy_score

class GaussianNaiveBayes:
    def __init__(self):
        """Initialize Gaussian Naive Bayes classifier"""
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        
    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize arrays to store parameters
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    def _calculate_likelihood(self, X, mean, var):
        """Calculate Gaussian likelihood"""
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((X - mean) ** 2) / (2 * var))
    
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
            Predicted labels
        """
        y_pred = []
        
        for x in X:
            posteriors = []
            
            # Calculate posterior probability for each class
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                likelihood = np.sum(np.log(self._calculate_likelihood(x, self.mean[idx], self.var[idx])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Get class with highest posterior probability
            y_pred.append(self.classes[np.argmax(posteriors)])
            
        return np.array(y_pred)
    
    def score(self, X, y):
        """
        Calculate accuracy
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            True labels
        
        Returns:
        --------
        accuracy : float
            Classification accuracy
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Initialize Multinomial Naive Bayes classifier
        
        Parameters:
        -----------
        alpha : float
            Smoothing parameter
        """
        self.alpha = alpha
        self.classes = None
        self.class_log_prior = None
        self.feature_log_prob = None
        
    def fit(self, X, y):
        """
        Train the Multinomial Naive Bayes model
        
        Parameters:
        -----------
        X : array-like
            Training features (count matrix)
        y : array-like
            Training labels
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Calculate class priors
        self.class_log_prior = np.zeros(n_classes)
        for idx, c in enumerate(self.classes):
            self.class_log_prior[idx] = np.log(np.sum(y == c) / n_samples)
        
        # Calculate feature probabilities
        self.feature_log_prob = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            count = np.sum(X_c, axis=0) + self.alpha
            total = np.sum(count)
            self.feature_log_prob[idx] = np.log(count / total)
    
    def predict(self, X):
        """
        Make predictions for new data
        
        Parameters:
        -----------
        X : array-like
            Features to predict (count matrix)
        
        Returns:
        --------
        predictions : array
            Predicted labels
        """
        y_pred = []
        
        for x in X:
            posteriors = []
            
            # Calculate posterior probability for each class
            for idx, c in enumerate(self.classes):
                prior = self.class_log_prior[idx]
                likelihood = np.sum(x * self.feature_log_prob[idx])
                posterior = prior + likelihood
                posteriors.append(posterior)
            
            # Get class with highest posterior probability
            y_pred.append(self.classes[np.argmax(posteriors)])
            
        return np.array(y_pred)
    
    def score(self, X, y):
        """
        Calculate accuracy
        
        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            True labels
        
        Returns:
        --------
        accuracy : float
            Classification accuracy
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred) 