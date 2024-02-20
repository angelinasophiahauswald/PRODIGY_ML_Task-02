import sys
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt

def read_data(data):
    df = pd.read_csv(data)
    return df

# One-Hot Encoding
def preprocessing_non_numericals(df):
    df['Gender'] = df['Gender'].replace({'Female': 1, 'Male': 0})
    return df

# Function to calculate distance between two data points
def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to assign points to clusters
def assign_clusters(X, centroids):
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        for j, point in enumerate(X):
            distances[j, i] = calculate_distance(centroid, point)
    return np.argmin(distances, axis=1)

# Function to update centroids
def update_centroids(X, labels, num_clusters):
    centroids = np.zeros((num_clusters, X.shape[1]))
    for i in range(num_clusters):
        centroids[i] = np.mean(X[labels == i], axis=0)
    return centroids

# Function to perform K-means clustering
def kmeans(X, num_clusters, max_iter=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], num_clusters, replace=False)]
    for _ in range(max_iter):
        # Assign points to the closest cluster centroid
        labels = assign_clusters(X, centroids)
        # Update centroids
        new_centroids = update_centroids(X, labels, num_clusters)
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Function to visualize clusters
def plot_clusters(X, centroids, labels, save_path=None):
    plt.figure(figsize=(8, 6))
    for i in range(len(centroids)):
        plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], label=f'Cluster {i}', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-means Clustering')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)  # Save the plot to a file
    else:
        plt.show()


    

if __name__ == '__main__':
    data = sys.argv[1]
    k = sys.argv[2]
    k = int(k)
    x = read_data(data)
    print(preprocessing_non_numericals(x))

    labels, centroids = kmeans(x.values, k)
    print(labels)
    plot_clusters(x.values, centroids, labels, save_path='clusters_plot.png')