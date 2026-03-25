import os
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def generate_mswep_precipitation_clusters():
    mswep_path = os.path.join("data.nosync", "mswep_1979_2020.nc")
    
    ds = xr.open_dataset(mswep_path)
    
    # Rename coordinates to match previous pipeline expectations
    ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    pr = ds['precipitation']
    
    # 0. Apply a strict geographical bounding box for the *whole* of Morocco
    # This prevents the deep Atlantic ocean from ruining the clusters.
    print("Slicing exactly to Morocco borders (Lat 36 down to 21, Lon -17 to -1)...")
    pr = pr.sel(latitude=slice(36.0, 21.0), longitude=slice(-17.0, -1.0))
    
    # 1. 3-day Rolling Window (Standard for CMM-VAE methodology)
    print("Applying 3-day rolling window to smooth precipitation...")
    pr_rolling = pr.rolling(time=3, min_periods=1, center=True).mean()
    
    # 2. Filter for Extended Winter (NDJFM: November to March)
    print("Filtering for extended winter months (NDJFM)...")
    extended_winter_months = [11, 12, 1, 2, 3]
    pr_winter = pr_rolling.sel(time=np.isin(pr_rolling.time.dt.month, extended_winter_months))
    
    # Fill any potential NaNs with 0
    pr_winter = pr_winter.fillna(0)
    
    print(f"Data preprocessed. Shape: {pr_winter.shape} (Time, Latitude, Longitude)")
    
    # 3. K-Means Clustering
    print("Reshaping data for KMeans clustering...")
    data_values = pr_winter.values
    nt, ny, nx = data_values.shape
    pr_reshaped = np.reshape(data_values, [nt, ny * nx], order='F')
    
    n_clusters = 5
    print(f"Running KMeans with k={n_clusters} (this may take a minute)...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    kmeans.fit(pr_reshaped)
    labels = kmeans.labels_
    
    # 4. Save Outputs
    output_dir = 'data.nosync/'
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Save cluster labels
    df_labels = pd.DataFrame({'labels': labels})
    out_labels = os.path.join(output_dir, 'MSWEP_pr_cluster_labels_5.csv')
    df_labels.to_csv(out_labels)
    print(f"Saved cluster labels to: {out_labels}")
    
    # B. Save total precipitation (extended winter)
    pr_winter.name = 'precipitation'
    out_total = os.path.join(output_dir, 'MSWEP_pr_total_EW.nc')
    pr_winter.to_netcdf(out_total)
    print(f"Saved total winter precipitation to: {out_total}")
    
    # C. Save 95th percentile exceedance (binary mask of Extreme Weather events)
    print("Calculating 95th percentile exceedance for extreme weather mapping...")
    pr_95_threshold = pr_winter.quantile(0.95, dim='time')
    pr_95_exceedance = xr.where(pr_winter > pr_95_threshold, 1, 0).astype(float)
    
    pr_95_exceedance.name = 'precipitation'
    out_95pc = os.path.join(output_dir, 'MSWEP_pr_95pc_EW.nc')
    pr_95_exceedance.to_netcdf(out_95pc)
    print(f"Saved 95th percentile mask to: {out_95pc}")
    
if __name__ == "__main__":
    generate_mswep_precipitation_clusters()
