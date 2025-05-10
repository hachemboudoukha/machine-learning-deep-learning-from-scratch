import numpy as np
from sklearn.datasets import load_iris, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score

from machine_learning.KNN.knn_model import KNN
from machine_learning.SVM.svm_model import SVM, KernelSVM
from machine_learning.Naive_Bayes.naive_bayes_model import GaussianNaiveBayes
from machine_learning.K_Means.k_means_model import KMeans

def main():
    print("Starting test_models.py...")
    
    # Test KNN
    print("\nTesting KNN...")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    knn_accuracy = knn.score(X_test, y_test)
    print(f"KNN accuracy: {knn_accuracy:.2f}")
    
    # Test SVM
    print("\nTesting SVM...")
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)
    svm_accuracy = svm.score(X_test, y_test)
    print(f"SVM accuracy: {svm_accuracy:.2f}")
    
    # Test Kernel SVM
    print("\nTesting Kernel SVM...")
    kernel_svm = KernelSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel='rbf', gamma=1.0)
    kernel_svm.fit(X_train, y_train)
    kernel_svm_accuracy = kernel_svm.score(X_test, y_test)
    print(f"Kernel SVM accuracy: {kernel_svm_accuracy:.2f}")
    
    # Test Gaussian Naive Bayes
    print("\nTesting Gaussian Naive Bayes...")
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    nb_accuracy = nb.score(X_test, y_test)
    print(f"Gaussian Naive Bayes score: {nb_accuracy:.2f}")
    
    # Test K-Means
    print("\nTesting K-Means...")
    X, _ = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, random_state=42)
    
    kmeans = KMeans(k=3, max_iters=100)
    kmeans.fit(X)
    kmeans_score = kmeans.score(X)
    print(f"K-Means silhouette score: {kmeans_score:.2f}")

if __name__ == "__main__":
    main() 