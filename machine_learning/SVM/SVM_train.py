import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from support_vector_machine import SVM, KernelSVM

# Generate synthetic data
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM
linear_svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
linear_svm.fit(X_train, y_train)
linear_predictions = linear_svm.predict(X_test)
linear_accuracy = accuracy_score(y_test, linear_predictions)
print(f"Linear SVM Accuracy: {linear_accuracy}")

# Kernel SVM
kernel_svm = KernelSVM(kernel='rbf', gamma=1.0, C=1.0)
kernel_svm.fit(X_train, y_train)
kernel_predictions = kernel_svm.predict(X_test)
kernel_accuracy = accuracy_score(y_test, kernel_predictions)
print(f"Kernel SVM Accuracy: {kernel_accuracy}")

# Visualization
plt.figure(figsize=(12, 5))

# Linear SVM plot
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=linear_predictions, cmap='viridis', marker='x', s=100, label='Predictions')
plt.title('Linear SVM')
plt.legend()

# Kernel SVM plot
plt.subplot(1, 2, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=kernel_predictions, cmap='viridis', marker='x', s=100, label='Predictions')
plt.title('Kernel SVM')
plt.legend()

plt.tight_layout()
plt.show() 