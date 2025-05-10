import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from k_nearest_neighbors import KNN

# Classification example
X, y = datasets.make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
knn = KNN(k=5, task='classification')
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Classification Accuracy: {accuracy}")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='viridis', marker='x', s=100, label='Predictions')
plt.title('KNN Classification')
plt.legend()
plt.show() 