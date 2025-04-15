"""
This python script, given a dataset and a count.
It will cluster the recipes using K-mediods clustering and find count number of represtative recipes.
"""

DATASET = "../datasets/sin.csv"
COUNT = 200
INITIAL_AREA = 5416

import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw

# ----------------------------
# 1. Load Data and Normalize
# ----------------------------

# Load the dataset
data = pd.read_csv(DATASET)

# List of AND count columns (AND1 to AND20)
and_columns = [f'AND{i}' for i in range(1, 21)]

# Normalize the AND counts by INITIAL_AREA
data[and_columns] = data[and_columns] / INITIAL_AREA

# -----------------------------
# 2. Prepare the Time Series Data
# -----------------------------

# Extract normalized AND data (shape: [n_samples, 20])
and_data = data[and_columns].values

# List of Level columns (Level1 to Level20)
level_columns = [f'Level{i}' for i in range(1, 21)]

# Extract level data (shape: [n_samples, 20])
level_data = data[level_columns].values

# Combine the two data sources to create a multivariate time series with 2 features per timestamp
# The resulting shape will be (n_samples, 20, 2) where the first channel is normalized AND and the second is Level.
time_series_data = np.dstack((and_data, level_data))

# -----------------------------
# 3. DTW-Based Time Series Clustering
# -----------------------------

n_clusters = COUNT

# Create a TimeSeriesKMeans model that uses DTW as distance metric
# Note: The DTW metric here allows for time warping in comparing the sequential data.
model = TimeSeriesKMeans(n_clusters=n_clusters, 
                         metric="dtw", 
                         max_iter=50, 
                         random_state=8, 
                         verbose=1, 
                         n_jobs=-1)

# Fit the clustering model and assign each recipe to a cluster
cluster_labels = model.fit_predict(time_series_data)

# Attach the cluster labels to the original dataframe for reference
data['Cluster'] = cluster_labels

# -----------------------------------------
# 4. Identify Key Representative Recipes (Medoids)
# -----------------------------------------

# For each cluster, we compute the DTW distance between each member’s time series and its cluster centroid.
# We then select the recipe with the smallest distance as the cluster’s representative.

representative_indices = []

for cluster in range(n_clusters):
    # Find the indices of recipes that belong to the current cluster
    cluster_indices = np.where(cluster_labels == cluster)[0]
    
    # Get the cluster's DTW centroid
    cluster_center = model.cluster_centers_[cluster]
    
    # Initialize variables to store the medoid for the current cluster
    min_distance = np.inf
    medoid_index = None
    
    # Loop over all recipes in the cluster to compute DTW distance to the centroid
    for idx in cluster_indices:
        distance = dtw(time_series_data[idx], cluster_center)
        if distance < min_distance:
            min_distance = distance
            medoid_index = idx
            
    # Save the index of the medoid for the current cluster
    representative_indices.append(medoid_index)

# -----------------------------------------
# 5. Save the Final Representative Recipes
# -----------------------------------------

# Extract the representative recipes (medoids) from the dataset using the stored indices.
representatives = data.iloc[representative_indices].drop(columns=level_columns)

# Save the final representatives to a CSV file named 'clustered.csv'
representatives.to_csv("clustered.csv", index=False)

print("Clustering completed and representative recipes saved to 'clustered.csv'.")
