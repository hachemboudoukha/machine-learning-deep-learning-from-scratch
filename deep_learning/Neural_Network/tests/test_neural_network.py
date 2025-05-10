import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the neural network module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.layers = [2, 3, 1]  # Simple network: 2 inputs, 3 hidden, 1 output
        self.nn = NeuralNetwork(self.layers, learning_rate=0.01)
        
        # Create simple XOR dataset
        self.X = np.array([[0, 0, 1, 1],
                          [0, 1, 0, 1]]).reshape(2, 4)
        self.y = np.array([[0, 1, 1, 0]]).reshape(1, 4)
    
    def test_initialization(self):
        """Test if the neural network is initialized correctly"""
        # Check if weights and biases are initialized
        self.assertEqual(len(self.nn.weights), len(self.layers)-1)
        self.assertEqual(len(self.nn.biases), len(self.layers)-1)
        
        # Check shapes of weights and biases
        self.assertEqual(self.nn.weights[0].shape, (3, 2))  # First layer
        self.assertEqual(self.nn.weights[1].shape, (1, 3))  # Second layer
        self.assertEqual(self.nn.biases[0].shape, (3, 1))   # First layer
        self.assertEqual(self.nn.biases[1].shape, (1, 1))   # Second layer
    
    def test_sigmoid(self):
        """Test sigmoid activation function"""
        # Test sigmoid at x = 0
        self.assertAlmostEqual(self.nn.sigmoid(0), 0.5)
        
        # Test sigmoid at large positive x
        self.assertAlmostEqual(self.nn.sigmoid(10), 0.9999546021312976)
        
        # Test sigmoid at large negative x
        self.assertAlmostEqual(self.nn.sigmoid(-10), 0.000045397868702423406)
    
    def test_sigmoid_derivative(self):
        """Test sigmoid derivative function"""
        # Test derivative at x = 0.5
        self.assertAlmostEqual(self.nn.sigmoid_derivative(0.5), 0.25)
        
        # Test derivative at x = 0
        self.assertAlmostEqual(self.nn.sigmoid_derivative(0), 0)
        
        # Test derivative at x = 1
        self.assertAlmostEqual(self.nn.sigmoid_derivative(1), 0)
    
    def test_forward_propagation(self):
        """Test forward propagation"""
        output = self.nn.forward(self.X)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 4))
        
        # Check if output is between 0 and 1 (sigmoid range)
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))
    
    def test_backward_propagation(self):
        """Test backward propagation"""
        # Perform forward pass
        self.nn.forward(self.X)
        
        # Store initial weights and biases
        initial_weights = [w.copy() for w in self.nn.weights]
        initial_biases = [b.copy() for b in self.nn.biases]
        
        # Perform backward pass
        self.nn.backward(self.X, self.y)
        
        # Check if weights and biases were updated
        for i in range(len(self.nn.weights)):
            self.assertFalse(np.array_equal(self.nn.weights[i], initial_weights[i]))
            self.assertFalse(np.array_equal(self.nn.biases[i], initial_biases[i]))
    
    def test_training(self):
        """Test training process"""
        # Train for a few epochs
        self.nn.train(self.X, self.y, epochs=100)
        
        # Make predictions
        predictions = self.nn.predict(self.X)
        
        # Check if predictions are binary (after thresholding)
        binary_predictions = (predictions > 0.5).astype(int)
        self.assertTrue(np.all(np.isin(binary_predictions, [0, 1])))
    
    def test_predict(self):
        """Test prediction method"""
        # Train the network
        self.nn.train(self.X, self.y, epochs=100)
        
        # Test prediction on training data
        predictions = self.nn.predict(self.X)
        
        # Check prediction shape
        self.assertEqual(predictions.shape, (1, 4))
        
        # Check if predictions are between 0 and 1
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))

if __name__ == '__main__':
    unittest.main() 