# Machine Learning & Deep Learning from Scratch

This project is a detailed implementation of Machine Learning and Deep Learning algorithms from scratch. The goal is to deeply understand the internal workings of these algorithms by implementing them without using high-level libraries (except for basic mathematical operations).

## 🚀 Features

### Implemented Machine Learning Algorithms
- [x] K-Means Clustering
- [x] Random Forest
- [x] Naive Bayes
- [x] K-Nearest Neighbors (KNN)
- [x] Support Vector Machine (SVM)
- [x] Linear Regression
- [x] Logistic Regression
- [x] Decision Tree

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/hachem-boudoukha/machine-learning-deep-learning-from-scratch.git
cd machine-learning-deep-learning-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Example with K-Means

```python
from machine_learning.K_Means.kmeans import KMeans
import numpy as np

# Create sample data
X = np.random.rand(100, 2)

# Initialize and train the model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Make predictions
predictions = kmeans.predict(X)
```

### Example with Random Forest

```python
from machine_learning.Random_Forest.random_forest import RandomForest
import numpy as np

# Create sample data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Initialize and train the model
rf = RandomForest(n_trees=10, max_depth=5)
rf.fit(X, y)

# Make predictions
predictions = rf.predict(X)
```

## 📁 Project Structure

```
machine-learning-deep-learning-from-scratch/
├── machine_learning/
│   ├── K_Means/
│   ├── Random_Forest/
│   ├── Naive_Bayes/
│   ├── KNN/
│   ├── SVM/
│   ├── Linear_Regression/
│   ├── Logistic_Regression/
│   └── Decision_Tree/
├── deep_learning/
├── data/
├── requirements.txt
└── setup.py
```

## 🧪 Testing

To run unit tests:

```bash
python test_models.py
```

## 📚 Documentation

Each algorithm is documented in its own directory with:
- Theoretical explanation of the algorithm
- Usage examples
- Detailed code comments

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 👥 Authors
- **Hachem (Mohamed El hachemi) Boudoukha**
  - GitHub: [@hachemboudoukha](https://github.com/hachemboudoukha)
  - LinkedIn: [Hachem Boudoukha](https://www.linkedin.com/in/hachem-boudoukha/)
  - Email: [hachemboudoukha@gmail.com](mailto:hachemboudoukha@gmail.com)

## 🙏 Acknowledgments

- Thanks to all contributors
- Inspired by machine learning specialization and Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition
