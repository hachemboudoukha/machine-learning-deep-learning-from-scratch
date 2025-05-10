import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        """
        Initialize neural network with given architecture
        layers: list of integers representing the number of neurons in each layer
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.cost_history = []
        
        # Initialize weights and biases
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i+1], layers[i]) * 0.01
            b = np.zeros((layers[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], self.activations[-1]) + self.biases[i]
            self.z_values.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward(self, X, Y):
        """Backward propagation"""
        m = X.shape[1]  # number of examples
        delta = self.activations[-1] - Y
        
        for i in range(len(self.weights)-1, -1, -1):
            # Compute gradients
            dW = np.dot(delta, self.activations[i].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # Compute delta for next layer
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(self.activations[i])
    
    def train(self, X, Y, epochs):
        """Train the neural network"""
        for i in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Compute cost
            cost = -np.mean(Y * np.log(output + 1e-8) + (1-Y) * np.log(1-output + 1e-8))
            self.cost_history.append(cost)
            
            # Backward propagation
            self.backward(X, Y)
            
            # Print cost every 1000 epochs
            if i % 1000 == 0:
                print(f"Cost after {i} epochs: {cost}")
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X) 