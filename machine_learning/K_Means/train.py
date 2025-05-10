import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from k_means_model import KMeans

def main():
    # Generate synthetic dataset
    X, _ = make_blobs(
        n_samples=300,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        random_state=42
    )
    
    # Initialize and train the model
    model = KMeans(k=3, max_iters=100)
    model.fit(X)
    
    # Get predictions
    labels = model.predict(X)
    
    # Calculate silhouette score
    score = model.score(X)
    
    print(f"K-Means Silhouette Score: {score:.2f}")

if __name__ == "__main__":
    main() 