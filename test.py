import torch
import numpy as np
from sklearn.cluster import KMeans

# Assume your data is stored in a PyTorch tensor 'data_tensor'
data_tensor = torch.tensor([[x1, x2, x3, ...], [y1, y2, y3, ...], ...])

# Convert the PyTorch tensor to a NumPy array
data_np = data_tensor.numpy()

# Specify the number of clusters (K)
K = 3

# Create a KMeans object and fit it to your data
kmeans = KMeans(n_clusters=K)
kmeans.fit(data_np)

# Get the cluster centers and labels
cluster_centers = torch.tensor(kmeans.cluster_centers_)
labels = torch.tensor(kmeans.labels_)

# Now you have the cluster centers and the assigned labels
