# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# setting display options for better readability
pd.set_option('display.max_columns', None)

# importing dataset 
df = pd.read_csv("Mall_Customers.csv")

# checking first few rows
print("Preview of dataset:")
print(df.head())

# selecting relevant columns based on project report
feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
df_model = df[feature_cols].dropna()

# displaying basic information
print("\nDataset shape:", df_model.shape)
print("\nChecking for missing values:")
print(df_model.isna().sum())

# exploring distributions of selected features
plt.figure()
df_model["Annual Income (k$)"].hist(bins=15)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Count")
plt.show()

plt.figure()
df_model["Spending Score (1-100)"].hist(bins=15)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Count")
plt.show()

# scaling features before clustering
scaler = StandardScaler()
X = scaler.fit_transform(df_model.values)

# executing elbow method to determine optimal cluster number
inertias = []
k_range = range(2, 11)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)

# plotting elbow curve
plt.figure()
plt.plot(list(k_range), inertias, marker="o")
plt.title("Elbow Method (Inertia vs k)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster SSE)")
plt.xticks(list(k_range))
plt.show()

# executing silhouette analysis (optional)
silhouette_scores = []
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.figure()
plt.plot(list(k_range), silhouette_scores, marker="o")
plt.title("Silhouette Score vs k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()

# executing final k-means model 
k_final = 6
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# adding cluster labels to original dataframe
df["Cluster"] = labels

# calculating cluster centers and converting back to original units
centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers, columns=feature_cols)
centers_df.index.name = "Cluster"

print("\nCluster Centers (Original Units):")
print(centers_df.round(2))

# visualizing clusters
plt.figure()
for c in range(k_final):
    mask = labels == c
    plt.scatter(df_model.values[mask, 0], df_model.values[mask, 1], s=35, label=f"Cluster {c}")
plt.scatter(centers[:, 0], centers[:, 1], s=150, marker="X", label="Centers")
plt.title("Customer Segments — Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# summarizing cluster sizes
cluster_sizes = df["Cluster"].value_counts().sort_index()
print("\nCluster Sizes:")
print(cluster_sizes)

plt.figure()
cluster_sizes.plot(kind="bar")
plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# exporting final dataset with cluster labels for reporting
df.to_csv("mall_customers_with_clusters.csv", index=False)
print("\nExported: mall_customers_with_clusters.csv")
