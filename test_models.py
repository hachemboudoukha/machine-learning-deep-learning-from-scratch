from machine_learning.KNN import KNN
from machine_learning.SVM import SVM, KernelSVM
from machine_learning.Naive_Bayes import GaussianNaiveBayes, MultinomialNaiveBayes
from machine_learning.K_Means import KMeans

import numpy as np
from sklearn.datasets import make_classification, make_blobs

# Test KNN
print("Testing KNN...")
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
knn = KNN(k=3)
knn.fit(X, y)
print("KNN score:", knn.score(X, y))

# Test SVM
print("\nTesting SVM...")
svm = SVM()
svm.fit(X, y)
print("SVM score:", svm.score(X, y))

# Test Kernel SVM
print("\nTesting Kernel SVM...")
kernel_svm = KernelSVM(kernel='rbf')
kernel_svm.fit(X, y)
print("Kernel SVM score:", kernel_svm.score(X, y))

# Test Gaussian Naive Bayes
print("\nTesting Gaussian Naive Bayes...")
gnb = GaussianNaiveBayes()
gnb.fit(X, y)
print("Gaussian Naive Bayes score:", gnb.score(X, y))

# Test Multinomial Naive Bayes (with non-negative data)
print("\nTesting Multinomial Naive Bayes...")
X_pos = np.abs(X)  # Make features non-negative for Multinomial NB
mnb = MultinomialNaiveBayes()
mnb.fit(X_pos, y)
print("Multinomial Naive Bayes score:", mnb.score(X_pos, y))

# Test K-Means
print("\nTesting K-Means...")
X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
print("K-Means silhouette score:", kmeans.score(X))
print("K-Means inertia:", kmeans.inertia(X)) 