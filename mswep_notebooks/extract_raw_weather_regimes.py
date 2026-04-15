

import os
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.cluster import KMeans

DATA_DIR = 'data.nosync'
RESULTS_DIR = 'predictability_paper-main/results'
OUTPUT_FILE = 'raw_weather_regimes_holdout.csv'

RANDOM_STATE = 42
HOLDOUT_FRAC = 0.3
GRAVITY_CONST = 9.80665

def main():
    # STEP 1: LOAD CALENDAR SPLIT
    pr_path = os.path.join(DATA_DIR, 'MSWEP_pr_total_DJF.nc')
    pr_spatial = xr.open_dataset(pr_path).precipitation
    times = pd.to_datetime(pr_spatial.time.values)

    winter_year = np.array([t.year if t.month == 12 else t.year - 1 for t in times])
    gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE)
    X_dummy = np.zeros((len(times), 1))
    train_idx, holdout_idx = next(gss.split(X_dummy, groups=winter_year))
    
    holdout_dates = times[holdout_idx]
    # STEP 2: RECONSTRUCT K-MEANS BASELINE Z500 WEATHER REGIMES

    print("Reconstructing the raw Z500 K-Means Weather Regimes")
    z500_path = os.path.join(DATA_DIR, 'era5_z500_daily_250_atlantic_1940_2022.nc')
    z500 = xr.open_dataset(z500_path).z.astype(np.float32) * (1.0 / GRAVITY_CONST)
    z500 = z500.sel(latitude=slice(20, 80), longitude=slice(-50, 30))
    z500 = z500.sel(time=np.isin(z500.time.dt.month, [12, 1, 2]))
    z500 = z500.where((z500['time.year'] >= 1979) & (z500['time.year'] <= 2020), drop=True)
    z500 = z500.groupby('time.dayofyear').map(lambda x: x - x.mean(dim='time'))
    z500 = z500.rolling(time=5, min_periods=1, center=True).mean()
    z500 = z500 * np.cos(np.deg2rad(z500.latitude))
    z500 = z500 / z500.std()

    z500_values = z500.values
    nt_z, ny_z, nx_z = z500_values.shape
    z500_flat = np.reshape(z500_values, [nt_z, ny_z * nx_z], order='F')
    
    kmeans_z = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
    kmeans_z.fit(z500_flat[train_idx])
    baseline_wr_raw = kmeans_z.predict(z500_flat)

    # STEP 3: EXTRACT CMM-VAE RAW WEATHER REGIMES
    print("Extracting CMM-VAE raw Weather Regimes...")
    c_enc_path = os.path.join(RESULTS_DIR, 'cmmvae_mswep_djf_wgroups/c_enc_wgroups.csv')
    df_cmm = pd.read_csv(c_enc_path)
    cmm_wr_raw = df_cmm['label'].values

    # STEP 4: EXPORT COMPARISON TABLE
    print("Generating pure Weather Regime table")

    # extract strictly the regimes without ANY mapping to rainfall clusters!
    # Adding +1 to make them 1-indexed (Regime 1, 2, 3, 4)
    final_baseline_wr = baseline_wr_raw[holdout_idx] + 1
    final_cmm_wr = cmm_wr_raw[holdout_idx] + 1

    df_final = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in holdout_dates],
        'KMeans_Z500_WR': final_baseline_wr,
        'CMM_VAE_WR': final_cmm_wr
    })

    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSUCCESS: Pure Weather Regimes Table generated at -> {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
