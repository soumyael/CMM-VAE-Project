"""
Regime Comparison Generator
---------------------------
This script reconstructs the 1,172 holdout days to compare the predictions 
of the CMM-VAE model against the traditional K-Means Baseline.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import KMeans
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = 'data.nosync'
RESULTS_DIR = 'predictability_paper-main/results'
OUTPUT_FILE = 'holdout_regime_comparison.csv'

RANDOM_STATE = 42
HOLDOUT_FRAC = 0.3
GRAVITY_CONST = 9.80665

def main():
    print("Starting Comparison Generation...\n")

    # =============================================================================
    # STEP 1: LOAD RAINFALL DATA & REPLICATE CALENDAR SPLIT
    # =============================================================================
    print("[1/5] Extracting study timeline from MSWEP dataset...")
    pr_path = os.path.join(DATA_DIR, 'MSWEP_pr_total_DJF.nc')
    pr_spatial = xr.open_dataset(pr_path).precipitation
    times = pd.to_datetime(pr_spatial.time.values)

    # I replicate the exact 'Winter-Year' split to isolate unseen holdout days
    winter_year = np.array([t.year if t.month == 12 else t.year - 1 for t in times])
    gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE)
    X_dummy = np.zeros((len(times), 1))
    train_idx, holdout_idx = next(gss.split(X_dummy, groups=winter_year))
    
    holdout_dates = times[holdout_idx]

    # =============================================================================
    # STEP 2: LOAD GROUND-TRUTH RAINFALL CLUSTERS
    # =============================================================================
    print("[2/5] Loading official MSWEP ground-truth rainfall patterns...")
    df_true = pd.read_csv(os.path.join(DATA_DIR, 'MSWEP_pr_cluster_labels_4_djf.csv'))
    true_labels_raw = df_true['labels'].values

    # =============================================================================
    # STEP 3: RECONSTRUCT K-MEANS BASELINE (ATMOSPHERE-ONLY)
    # =============================================================================
    print("[3/5] Reconstructing the Z500 K-Means Baseline...")
    
    # I preprocess the ERA5 data exactly as modeled in the main notebooks
    z500_path = os.path.join(DATA_DIR, 'era5_z500_daily_250_atlantic_1940_2022.nc')
    z500 = xr.open_dataset(z500_path).z.astype(np.float32) * (1.0 / GRAVITY_CONST)
    z500 = z500.sel(latitude=slice(20, 80), longitude=slice(-50, 30))
    z500 = z500.sel(time=np.isin(z500.time.dt.month, [12, 1, 2]))
    z500 = z500.where((z500['time.year'] >= 1979) & (z500['time.year'] <= 2020), drop=True)
    z500 = z500.groupby('time.dayofyear').map(lambda x: x - x.mean(dim='time'))
    z500 = z500.rolling(time=5, min_periods=1, center=True).mean()
    z500 = z500 * np.cos(np.deg2rad(z500.latitude))
    z500 = z500 / z500.std()

    # Train KMeans on training data only to prevent leakage
    z500_values = z500.values
    nt_z, ny_z, nx_z = z500_values.shape
    z500_flat = np.reshape(z500_values, [nt_z, ny_z * nx_z], order='F')
    
    kmeans_z = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
    kmeans_z.fit(z500_flat[train_idx])
    baseline_z_raw = kmeans_z.predict(z500_flat)

    # =============================================================================
    # STEP 4: EXTRACT CMM-VAE NEURAL NETWORK PREDICTIONS
    # =============================================================================
    print("[4/5] Extracting CMM-VAE probability predictions...")
    c_enc_path = os.path.join(RESULTS_DIR, 'cmmvae_mswep_djf_wgroups/c_enc_wgroups.csv')
    df_cmm = pd.read_csv(c_enc_path)
    cmm_regimes_raw = df_cmm['label'].values

    # =============================================================================
    # STEP 5: ALIGN, MAP, AND EXPORT COMPARISON TABLE
    # =============================================================================
    print("[5/5] Aligning results and saving table...")

    # I map the machine labels back to the corresponding human-readable rainfall clusters
    def map_to_human_clusters(predicted_labels, true_labels, train_mask):
        mapping_dict = {}
        p_train = predicted_labels[train_mask]
        t_train = true_labels[train_mask]
        
        for cluster_id in range(4):
            # Find all days where this cluster was predicted during training
            mask = (p_train == cluster_id)
            if np.any(mask):
                # I identify the most common actual rainfall pattern for those days
                mode_val = stats.mode(t_train[mask], keepdims=True).mode[0]
                mapping_dict[cluster_id] = int(mode_val)
            else:
                mapping_dict[cluster_id] = cluster_id
                
        # Apply the mapping and convert to 1-indexed (1, 2, 3, 4)
        return np.array([mapping_dict[val] + 1 for val in predicted_labels])

    # Extract answers solely for the 1,172 unseen holdout days
    final_true = true_labels_raw[holdout_idx] + 1
    final_cmm = map_to_human_clusters(cmm_regimes_raw, true_labels_raw, train_idx)[holdout_idx]
    final_base = map_to_human_clusters(baseline_z_raw, true_labels_raw, train_idx)[holdout_idx]

    # Create the final clean Pandas DataFrame
    df_final = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in holdout_dates],
        'True_Rain_Cluster': final_true.astype(int),
        'CMM_VAE_Regime': final_cmm,
        'Baseline_Regime': final_base
    })

    # Save to disk
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSUCCESS: Comparison Table generated at -> {OUTPUT_FILE}")
    print("=" * 60)
    
    # Print Quick Verification
    from sklearn.metrics import accuracy_score
    acc_cmm = accuracy_score(df_final['True_Rain_Cluster'], df_final['CMM_VAE_Regime'])
    acc_base = accuracy_score(df_final['True_Rain_Cluster'], df_final['Baseline_Regime'])
    print(f"CMM-VAE Holdout Accuracy:  {acc_cmm:.1%}")
    print(f"Baseline Holdout Accuracy: {acc_base:.1%}")

if __name__ == "__main__":
    main()
