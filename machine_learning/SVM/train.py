import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from svm_model import SVM, KernelSVM

def main():
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_classes=2,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train and evaluate linear SVM
    linear_svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    linear_svm.fit(X_train, y_train)
    linear_accuracy = linear_svm.score(X_test, y_test)
    print(f"Linear SVM Accuracy: {linear_accuracy:.2f}")
    
    # Train and evaluate kernel SVM
    kernel_svm = KernelSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='rbf', gamma=1.0)
    kernel_svm.fit(X_train, y_train)
    kernel_accuracy = kernel_svm.score(X_test, y_test)
    print(f"Kernel SVM Accuracy: {kernel_accuracy:.2f}")

if __name__ == "__main__":
    main() 