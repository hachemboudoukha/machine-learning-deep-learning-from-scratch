# Machine Learning Algorithms from Scratch

This directory contains implementations of various machine learning algorithms from scratch using only NumPy. The goal is to understand the fundamental concepts and mathematics behind these algorithms.

## 🧮 Implemented Algorithms

1. **K-Means Clustering**
   - Unsupervised learning
   - Clustering algorithm
   - Centroid-based clustering

2. **Random Forest**
   - Ensemble learning
   - Multiple decision trees
   - Classification and regression

3. **Naive Bayes**
   - Probabilistic classifier
   - Based on Bayes' theorem
   - Text classification

4. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Non-parametric method
   - Classification and regression

5. **Support Vector Machine (SVM)**
   - Linear and non-linear classification
   - Margin maximization
   - Kernel trick

6. **Linear Regression**
   - Simple linear regression
   - Multiple linear regression
   - Least squares method

7. **Logistic Regression**
   - Binary classification
   - Sigmoid function
   - Maximum likelihood estimation

8. **Decision Tree**
   - Tree-based model
   - Information gain
   - Classification and regression

## 📋 Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn (for testing and comparison)

## 🛠️ Usage

Each algorithm is implemented in its own directory with:
- Algorithm implementation file
- Example usage
- Documentation

### Example: Using K-Means

```python
from K_Means.kmeans import KMeans
import numpy as np

# Create sample data
X = np.random.rand(100, 2)

# Initialize and train the model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Make predictions
predictions = kmeans.predict(X)
```

## 📚 Documentation

Each algorithm directory contains:
- Theoretical explanation
- Implementation details
- Usage examples
- Performance metrics

## 🧪 Testing

Each algorithm includes:
- Unit tests
- Example usage scripts
- Performance comparison with scikit-learn

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the project
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request 