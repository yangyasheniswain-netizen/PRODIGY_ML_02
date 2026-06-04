import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Create K-Means model
kmeans = KMeans(n_clusters=5, random_state=42)

# Train the model
kmeans.fit(X)

# Assign cluster labels
df['Cluster'] = kmeans.labels_

# Plot customer clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'])

# Plot cluster centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()