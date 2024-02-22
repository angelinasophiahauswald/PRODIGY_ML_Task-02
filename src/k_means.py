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

# Function to calculate euclidean distance between two data points
def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Function to assign points to nearest cluster
def assign_clusters(X, centroids):
    distances = np.zeros((X.shape[0], centroids.shape[0]))
    x_index = 0
    for data_point in X:
        centroid_index = 0
        for centroid in centroids:
            distance = calculate_distance(centroid, data_point)
            distances[x_index, centroid_index] = distance
            centroid_index = centroid_index + 1
        x_index = x_index + 1
    return np.argmin(distances, axis=1)

# Function to update centroids
def update_centroids(X, cluster, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(X[cluster == i], axis=0)
    return centroids

# Function to perform K-means clustering
def k_means(X, k, iteration):
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for i in range(iteration):
        # Assign points to the closest cluster centroid
        cluster = assign_clusters(X, centroids)
        # Update centroids
        new_centroids = update_centroids(X, cluster, k)
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return cluster, centroids

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
    preprocessing_non_numericals(x)
    cluster, centroids = k_means(x.values, k, 100)
    plot_clusters(x.values, centroids, cluster, save_path='clusters_plot.png')