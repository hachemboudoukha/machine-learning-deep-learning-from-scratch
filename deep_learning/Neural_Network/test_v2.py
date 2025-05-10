import numpy as np
from neural_network_v2 import NeuralNetwork

def main():
    # Create a simple XOR dataset
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    # Define neural network architecture
    layers = [2, 4, 1]  # 2 inputs, 4 hidden neurons, 1 output
    
    # Initialize and train the model
    nn = NeuralNetwork(layers, learning_rate=0.1)
    
    # Train for 10000 epochs
    nn.train(X, y, epochs=10000)
    
    # Make predictions
    predictions = nn.predict(X)
    print("\nPredictions:")
    print(predictions)
    
    # Print final cost
    print(f"\nFinal cost: {nn.cost_history[-1]:.6f}")

if __name__ == "__main__":
    main() 