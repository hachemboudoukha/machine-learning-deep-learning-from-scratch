import numpy as np
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize SVM classifier
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        lambda_param : float
            Regularization parameter
        n_iters : int
            Number of iterations for training
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def _initialize_weights(self, n_features):
        """Initialize weights and bias"""
        self.w = np.zeros(n_features)
        self.b = 0
        
    def _get_hinge_loss(self, X, y):
        """Calculate hinge loss"""
        y_pred = np.dot(X, self.w) + self.b
        return np.maximum(0, 1 - y * y_pred)
    
    def _get_gradients(self, X, y):
        """Calculate gradients for gradient descent"""
        y_pred = np.dot(X, self.w) + self.b
        mask = y * y_pred < 1
        
        dw = self.lambda_param * self.w - np.sum(X[mask] * y[mask, np.newaxis], axis=0)
        db = -np.sum(y[mask])
        
        return dw, db
    
    def fit(self, X, y):
        """
        Train the SVM model
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels (-1 or 1)
        """
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self._initialize_weights(n_features)
        
        for _ in range(self.n_iters):
            dw, db = self._get_gradients(X, y_)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
    
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
            Predicted labels (0 or 1)
        """
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
    
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

class KernelSVM:
    def __init__(self, kernel='rbf', gamma=1.0, C=1.0):
        """
        Initialize Kernel SVM classifier
        
        Parameters:
        -----------
        kernel : str
            Type of kernel ('rbf' or 'linear')
        gamma : float
            Kernel coefficient for RBF kernel
        C : float
            Regularization parameter
        """
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
        
    def _rbf_kernel(self, x1, x2):
        """Calculate RBF kernel"""
        return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
    
    def _linear_kernel(self, x1, x2):
        """Calculate linear kernel"""
        return np.dot(x1, x2)
    
    def _get_kernel(self, x1, x2):
        """Get kernel value based on kernel type"""
        if self.kernel == 'rbf':
            return self._rbf_kernel(x1, x2)
        return self._linear_kernel(x1, x2)
    
    def fit(self, X, y):
        """
        Train the Kernel SVM model
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels (-1 or 1)
        """
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        
        # Initialize dual variables
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Calculate kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._get_kernel(X[i], X[j])
        
        # SMO algorithm (simplified version)
        for _ in range(100):  # Number of iterations
            for i in range(n_samples):
                Ei = np.sum(self.alpha * y * K[i, :]) + self.b - y[i]
                
                if (y[i] * Ei < -1e-3 and self.alpha[i] < self.C) or \
                   (y[i] * Ei > 1e-3 and self.alpha[i] > 0):
                    
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    Ej = np.sum(self.alpha * y * K[j, :]) + self.b - y[j]
                    
                    # Update alpha_i and alpha_j
                    old_alpha_i = self.alpha[i]
                    old_alpha_j = self.alpha[j]
                    
                    L = max(0, old_alpha_j + old_alpha_i - self.C)
                    H = min(self.C, old_alpha_j + old_alpha_i)
                    
                    if L == H:
                        continue
                    
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    self.alpha[j] = old_alpha_j - y[j] * (Ei - Ej) / eta
                    self.alpha[j] = max(L, min(H, self.alpha[j]))
                    
                    if abs(self.alpha[j] - old_alpha_j) < 1e-4:
                        continue
                    
                    self.alpha[i] = old_alpha_i + y[i] * y[j] * (old_alpha_j - self.alpha[j])
                    
                    # Update bias
                    b1 = self.b - Ei - y[i] * (self.alpha[i] - old_alpha_i) * K[i, i] - \
                         y[j] * (self.alpha[j] - old_alpha_j) * K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alpha[i] - old_alpha_i) * K[i, j] - \
                         y[j] * (self.alpha[j] - old_alpha_j) * K[j, j]
                    self.b = (b1 + b2) / 2
    
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
            Predicted labels (-1 or 1)
        """
        predictions = []
        for x in X:
            prediction = 0
            for i in range(len(self.X_train)):
                prediction += self.alpha[i] * self.y_train[i] * self._get_kernel(x, self.X_train[i])
            predictions.append(np.sign(prediction + self.b))
        return np.array(predictions)
    
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