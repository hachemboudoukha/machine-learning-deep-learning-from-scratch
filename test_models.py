print('Starting test_models.py...')
from machine_learning.KNN.k_nearest_neighbors import KNN
from machine_learning.SVM.support_vector_machine import SVM, KernelSVM
from machine_learning.Naive_Bayes.gaussian_naive_bayes import GaussianNaiveBayes
from machine_learning.K_Means.k_means_clustering import KMeans

import numpy as np
from sklearn.datasets import make_classification, make_blobs

# Test KNN
print("Testing KNN...")
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
knn = KNN(k=3)
knn.fit(X, y)
pred_knn = knn.predict(X)
accuracy_knn = np.mean(pred_knn == y)
print("KNN accuracy:", accuracy_knn)

# Test SVM
print("\nTesting SVM...")
svm = SVM()
svm.fit(X, y)
pred_svm = svm.predict(X)
accuracy_svm = np.mean(pred_svm == y)
print("SVM accuracy:", accuracy_svm)

# Test Kernel SVM
print("\nTesting Kernel SVM...")
kernel_svm = KernelSVM(kernel='rbf')
kernel_svm.fit(X, y)
pred_kernel_svm = kernel_svm.predict(X)
accuracy_kernel_svm = np.mean(pred_kernel_svm == y)
print("Kernel SVM accuracy:", accuracy_kernel_svm)

# Test Gaussian Naive Bayes
print("\nTesting Gaussian Naive Bayes...")
gnb = GaussianNaiveBayes()
gnb.fit(X, y)
print("Gaussian Naive Bayes score:", gnb.score(X, y))

# Test K-Means
print("\nTesting K-Means...")
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("K-Means silhouette score:", kmeans.score(X))
print("K-Means inertia:", kmeans.inertia(X)) 