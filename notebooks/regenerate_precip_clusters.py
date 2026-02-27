import sys
import os

# Add parent directory to import utility functions
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except:
    pass

import xarray as xr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import preprocessing_functions as pf
import clustering_functions as cf

def generate_precipitation_clusters():
    
    # Define dataset path (try multiple possible locations)
    filename = 'data_publication/CHIRPS_morocco_1980_2022.nc'
    
    if not os.path.exists(filename):
        filename = '../data_publication/CHIRPS_morocco_1980_2022.nc'
    if not os.path.exists(filename):
        filename = '../../data_publication/CHIRPS_morocco_1980_2022.nc'
    if not os.path.exists(filename):
        filename = 'C:/Users/Lenovo/Desktop/CMM-VAE-Morocco3/data_publication/CHIRPS_morocco_1980_2022.nc'
    
    print(f"Loading and preprocessing CHIRPS data from {filename}...")
    
    try:
        ds = xr.open_dataset(filename)
        var_name = list(ds.data_vars)[0]
        
        # Check latitude order to avoid slicing errors
        geo_filter = 'morocco'
        if ds.latitude.values[0] < ds.latitude.values[-1]:
            print("Latitude ascending → no strict geographical slicing.")
            geo_filter = None
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Preprocess precipitation data (3-day rolling average)
    pr = pf.preprocess_dataset(
        filename=filename,
        variable_name=var_name,
        multiplication_factor=1,
        geographical_filter=geo_filter,
        months_filter=extended_winter_months,
        anomalies=False,
        normalization=False,
        rolling_window=3
    )

    # Keep years before 2023
    pr = pr.where(pr['time.year'] < 2023, drop=True)

    print("Filling NaNs with 0...")
    pr = pr.fillna(0)
    
    print(f"Data preprocessed. Shape: {pr.shape}")
    
    # Load data again for clustering (safer preprocessing)
    print("Preparing data for clustering...")
    
    pr_clustering_data = pf.preprocess_dataset(
        filename=filename,
        variable_name=var_name,
        multiplication_factor=1,
        geographical_filter=geo_filter,
        months_filter=extended_winter_months,
        anomalies=False,
        normalization=False,
        rolling_window=3
    )

    pr_clustering_data = pr_clustering_data.where(
        pr_clustering_data['time.year'] < 2023, drop=True
    )
    pr_clustering_data = pr_clustering_data.fillna(0)

    print(f"Clustering data shape: {pr_clustering_data.shape}")
    print("Reshaping data for KMeans...")
    
    # Reshape to 2D (samples × features)
    pr_reshaped = cf.reshape_data_for_clustering(pr_clustering_data)
    
    # Apply KMeans clustering
    n_clusters = 5
    print(f"Running KMeans with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    kmeans.fit(pr_reshaped)
    labels = kmeans.labels_
    
    # Prepare output directory
    output_dir = 'data.nosync/'
    if not os.path.exists('data.nosync'):
        output_dir = '../data.nosync/'
    if not os.path.exists(output_dir):
        os.makedirs('data.nosync', exist_ok=True)
        output_dir = 'data.nosync/'
    
    # 1️ Save cluster labels
    df_labels = pd.DataFrame({'labels': labels})
    df_labels.to_csv(output_dir + 'CHIRPS_pr_cluster_labels_5.csv')
    print(f"Saved {output_dir}CHIRPS_pr_cluster_labels_5.csv")

    # 2️ Save total precipitation (extended winter)
    print(f"Saving {output_dir}CHIRPS_pr_total_EW.nc...")
    pr.name = 'precipitation_amount'
    pr.to_netcdf(output_dir + 'CHIRPS_pr_total_EW.nc')

    # 3️ Save 95th percentile exceedance (binary 0/1)
    print("Calculating 95th percentile exceedance indicator...")
    
    # Compute 95th percentile threshold (per grid point)
    pr_95_threshold = pr.quantile(0.95, dim='time')
    
    # Create binary indicator: 1 if pr > threshold, else 0
    pr_95_exceedance = xr.where(pr > pr_95_threshold, 1, 0).astype(float)
    
    unique_vals = np.unique(pr_95_exceedance.values)
    print(f"Unique values: {unique_vals} (should be [0. 1.])")
    print(f"Exceedance fraction: {pr_95_exceedance.values.mean():.4f}")
    
    pr_95_exceedance.name = 'precipitation_amount'
    pr_95_exceedance.to_netcdf(output_dir + 'CHIRPS_pr_95pc_EW.nc')
    print(f"Saved {output_dir}CHIRPS_pr_95pc_EW.nc (Shape: {pr_95_exceedance.shape})")


if __name__ == "__main__":
    extended_winter_months = [11, 12, 1, 2, 3]  # NDJFM
    generate_precipitation_clusters()