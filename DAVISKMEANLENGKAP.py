import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Sample data input
file_path = 'Downloads/impute/prodluas.csv'  # Update the file path if necessary
df = pd.read_csv(file_path)

# Select features for clustering
X = df[['Produksi (Ton)', 'Luas (ha)']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Elbow Method to find the optimal number of clusters
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method for Optimal k')
plt.show()

# Calculate Davies-Bouldin Index for different k values
dbi = []
for k in k_range[1:]:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    labels = kmeans.labels_
    dbi.append(davies_bouldin_score(X_scaled, labels))

# Plot the Davies-Bouldin Index
plt.figure(figsize=(10, 6))
plt.plot(k_range[1:], dbi, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index for Different k values')
plt.show()

# Display the optimal number of clusters according to Elbow and DBI
optimal_k_elbow = np.argmin(np.diff(sse)) + 1
optimal_k_dbi = np.argmin(dbi) + 2  # Add 2 because index starts from 1 and we skip the first value

print("Optimal k according to Elbow Method:", optimal_k_elbow)
print("Optimal k according to Davies-Bouldin Index:", optimal_k_dbi)

# Perform KMean clustering with the optimal number of clusters
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Produksi (Ton)'], df['Luas (ha)'], c=df['Cluster'], cmap='viridis', marker='o')
plt.xlabel('Produksi (Ton)')
plt.ylabel('Luas (ha)')
plt.title('2 Cluster, Total Produksi dan Luas Kabupaten dari 5 Provinsi Terpilih Periode 2006-2023 (KMeans)')

# Denormalize centroids for plotting
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.legend()
plt.show()

# Print the dataframe with cluster assignments
print(df)

# Print centroids
print("Cluster centroids (denormalized):")
print(centroids)

# Perform KMedoids clustering
optimal_k = 2
kmedoids = KMedoids(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmedoids.fit_predict(X_scaled)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Produksi (Ton)'], df['Luas (ha)'], c=df['Cluster'], cmap='viridis', marker='o')
plt.xlabel('Produksi (Ton)')
plt.ylabel('Luas (ha)')
plt.title('2 Cluster, Total Produksi dan Luas Kabupaten dari 5 Provinsi Terpilih Periode 2006-2023 (KMedoids)')

# Plot medoids
medoids = scaler.inverse_transform(kmedoids.cluster_centers_)
plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='X', s=200, label='Medoids')

plt.legend()
plt.show()

# Print the dataframe with cluster assignments
print(df)

# Print medoids
print("Cluster medoids (denormalized):")
print(medoids)