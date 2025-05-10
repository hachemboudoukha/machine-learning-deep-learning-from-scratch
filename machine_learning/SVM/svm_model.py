import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

class KernelSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='rbf', gamma=1.0):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.gamma = gamma
        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None
        
    def rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = np.where(y <= 0, -1, 1)
        
        # Initialize weights and bias
        self.w = np.zeros(n_samples)
        self.b = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            for i in range(n_samples):
                if self.kernel == 'rbf':
                    kernel_values = np.array([self.rbf_kernel(X[i], x) for x in X])
                else:
                    kernel_values = np.dot(X, X[i])
                
                condition = self.y_train[i] * (np.sum(self.w * kernel_values) - self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - kernel_values * self.y_train[i])
                    self.b -= self.lr * self.y_train[i]
    
    def predict(self, X):
        predictions = []
        for x in X:
            if self.kernel == 'rbf':
                kernel_values = np.array([self.rbf_kernel(x, x_train) for x_train in self.X_train])
            else:
                kernel_values = np.dot(X, x)
            
            prediction = np.sign(np.sum(self.w * kernel_values) - self.b)
            predictions.append(prediction)
        return np.array(predictions)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) 