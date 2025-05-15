# Deep Learning from Scratch

This directory contains implementations of various deep learning models from scratch using only NumPy. The goal is to understand the fundamental concepts and mathematics behind these models.

## 🧠 Implemented Models

1. **Neural Network (NN)**
   - Basic feedforward neural network
   - Backpropagation
   - Sigmoid activation
   - Binary classification
TO DO LATER :
2. **Convolutional Neural Network (CNN)**
   - Convolutional layers
   - Pooling layers
   - Fully connected layers
   - Image classification

3. **Recurrent Neural Network (RNN)**
   - Simple RNN cells
   - Sequence processing
   - Time series prediction

4. **Long Short-Term Memory (LSTM)**
   - LSTM cells
   - Long-term dependency learning
   - Sequence modeling

5. **Autoencoder**
   - Encoder-decoder architecture
   - Dimensionality reduction
   - Feature learning

6. **Generative Adversarial Network (GAN)**
   - Generator network
   - Discriminator network
   - Image generation

7. **Transformer**
   - Self-attention mechanism
   - Multi-head attention
   - Position-wise feedforward networks

8. **Variational Autoencoder (VAE)**
   - Probabilistic encoder
   - Probabilistic decoder
   - Latent space representation

## 📋 Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn

## 🛠️ Usage

Each model is implemented in its own directory with:
- Model implementation file (e.g., `neural_network.py`)
- Training script (e.g., `NN_train.py`)
- Example usage and documentation

### Example: Training a Neural Network

```python
from neural_network import NeuralNetwork

# Create and train the model
nn = NeuralNetwork(layers=[20, 10, 5, 1])
nn.train(X_train, y_train, epochs=5000)

# Make predictions
predictions = nn.predict(X_test)
```

## 📚 Documentation

Each model directory contains:
- Theoretical explanation
- Implementation details
- Usage examples
- Training scripts

## 🧪 Testing

Each model includes:
- Unit tests
- Example training scripts
- Performance metrics

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request 
