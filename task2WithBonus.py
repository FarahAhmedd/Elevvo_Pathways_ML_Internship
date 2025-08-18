from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
# Load data
data = pd.read_csv("Datasets/Mall_Customers.csv")
# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visual exploration
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Distribution')
plt.show()

# Determine optimal number of clusters using Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# Apply KMeans with optimal clusters (elbow point k=5)
k_opt = 5
kmeans = KMeans(n_clusters=k_opt, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data['Cluster'] = clusters

# Visualize clusters
plt.figure(figsize=(8,6))
for cluster in range(k_opt):
    plt.scatter(
        data[data['Cluster'] == cluster]['Annual Income (k$)'],
        data[data['Cluster'] == cluster]['Spending Score (1-100)'],
        label=f'Cluster {cluster}'
    )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.legend()
plt.show()

# Apply Gaussian Mixture Model (GMM) clustering
gmm = GaussianMixture(n_components=k_opt, random_state=42)
gmm_clusters = gmm.fit_predict(X_scaled)
data['GMM_Cluster'] = gmm_clusters

# Visualize GMM clusters
plt.figure(figsize=(8,6))
for cluster in range(k_opt):
    plt.scatter(
        data[data['GMM_Cluster'] == cluster]['Annual Income (k$)'],
        data[data['GMM_Cluster'] == cluster]['Spending Score (1-100)'],
        label=f'GMM Cluster {cluster}'
    )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments (GMM)')
plt.legend()
plt.show()

# Analyze average spending per cluster for KMeans
print("KMeans average spending per cluster:")
print(data.groupby('Cluster')['Spending Score (1-100)'].mean())

# Analyze average spending per cluster for GMM
print("\nGMM average spending per cluster:")
print(data.groupby('GMM_Cluster')['Spending Score (1-100)'].mean())