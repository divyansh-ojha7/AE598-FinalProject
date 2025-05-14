#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from sklearn.preprocessing import MinMaxScaler

# ─── 1) Load & clean data ─────────────────────────────────────────────
CSV_PATH = r'C:\Users\allyi\OneDrive\Desktop\SC Rentry\2021-05-26 DMPA_compiled_data.csv'  # CHANGE TO YOUR OWN FILE LOCATION

df = pd.read_csv(CSV_PATH)

# Columns
dens_col = 'FreestreamH'
vel_col  = 'Freestream velocity, m/s'
B0_col   = 'Magnet strength, T'
rc_col   = 'Magnet radius, m'
cdA_col  = 'Effective CD*A, m^2'


df = df.dropna(subset=[dens_col, vel_col, B0_col, rc_col, cdA_col])
# ─── 2) Assemble full arrays ────────────────────────────────────────────────
X = df[[dens_col, vel_col, B0_col, rc_col]].values
Y = df[cdA_col].values

# ─── 3) Normalize on the *full* dataset ────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)                  # fit on all rows
orig_min = scaler.data_min_    # true min per column
orig_max = scaler.data_max_    # true max per column

# ─── 4) Random subsample for interpolator ─────────────────────────────────
n_samples = 2000               # pick however many points you can afford
rng = np.random.default_rng(42)
idx = rng.choice(X.shape[0], size=n_samples, replace=False)

X_small = X[idx]
Y_small = Y[idx]

# scale your small cloud
X_small_scaled = scaler.transform(X_small)

# build interpolators
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
interp_lin = LinearNDInterpolator(X_small_scaled, Y_small, fill_value=np.nan)
interp_nn  = NearestNDInterpolator( X_small_scaled, Y_small)

# ─── 5) Estimator uses true min/max and scaler ─────────────────────────────
def estimate_cdA(density, velocity, B0, rc):
    pt = np.array([[density, velocity, B0, rc]])
    # clamp using *true* orig_min/orig_max
    pt_clamped = np.clip(pt, orig_min, orig_max)
    pt_scaled  = scaler.transform(pt_clamped)
    val = interp_lin(pt_scaled)[0]
    if np.isnan(val):
        return float(interp_nn(pt_scaled)[0])
    return float(val)

# ─── 6) User interface ────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== CD*A Estimator ===")
    # Show valid ranges for each input
    print("Valid input ranges:")
    print(f"  Density: {orig_min[0]:.3e} to {orig_max[0]:.3e} #/m³")
    print(f"  Velocity: {orig_min[1]:.3e} to {orig_max[1]:.3e} m/s")
    print(f"  Magnet strength (B0): {orig_min[2]:.3e} to {orig_max[2]:.3e} T")
    print(f"  Coil radius: {orig_min[3]:.3e} to {orig_max[3]:.3e} m")

    d = float(input("Enter freestream density (#/m³): "))
    u = float(input("Enter entry velocity (m/s):      "))
    b = float(input("Enter magnet strength (T):       "))
    r = float(input("Enter coil radius (m):            "))

    cdA = estimate_cdA(d, u, b, r)
    if np.isnan(cdA):
        print("\nOutside interpolation result (nan). Try different values within range.")
    else:
        print(f"\n⇒ Estimated CD*A = {cdA:.6f} m²")
