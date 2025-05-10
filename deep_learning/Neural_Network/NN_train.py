import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from neural_network_v2 import NeuralNetwork

def main():
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Reshape y for neural network (1 x n_samples)
    y = y.reshape(1, -1)
    
    # Split the data (X is n_samples x n_features, y is 1 x n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.T, test_size=0.2, random_state=42
    )
    
    # Transpose X for neural network (n_features x n_samples)
    X_train = X_train.T
    X_test = X_test.T
    
    # Define neural network architecture
    layers = [20, 10, 5, 1]  # Input layer: 20 features, 2 hidden layers, output layer: 1 neuron
    
    # Initialize and train the model
    nn = NeuralNetwork(layers, learning_rate=0.01)
    nn.train(X_train, y_train, epochs=5000)
    
    # Make predictions
    train_predictions = (nn.predict(X_train) > 0.5).astype(int)
    test_predictions = (nn.predict(X_test) > 0.5).astype(int)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train.T, train_predictions.T)
    test_accuracy = accuracy_score(y_test.T, test_predictions.T)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(nn.cost_history)
    plt.title('Training Cost Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.savefig('neural_network_training.png')
    plt.close()

if __name__ == "__main__":
    main() 