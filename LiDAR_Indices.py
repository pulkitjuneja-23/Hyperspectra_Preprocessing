"""
LiDAR Indices Calculation
Computes LiDAR-derived indices from raw measurements without standardization.
"""

import numpy as np
import pandas as pd
import os

# User Input
data_file = input("CSV/Excel file path: ").strip().strip('"').strip("'")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"File not found: {data_file}")

base_name = os.path.splitext(data_file)[0]
default_output = f"{base_name}_with_indices.csv"
output_file = input(f"Output CSV path (default: {default_output}): ").strip().strip('"').strip("'")
if not output_file:
    output_file = default_output
elif not output_file.endswith('.csv'):
    output_file = f"{output_file}.csv"

# Load Data
df = pd.read_csv(data_file) if data_file.endswith('.csv') else pd.read_excel(data_file)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Verify required columns
required_cols = ["Rall", "Rgrd", "Rveg", "veg 1st", "veg same", "veg 2nd", 
                 "grd same", "grd 1st", "grd 2nd", "all 1st", "all same", "all 2nd"]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Helper Function
def safe_div(num, den):
    """Safe division with NaN for zero/invalid denominators."""
    return np.where(den > 0, num / den, np.nan)

# Compute Indices
k = 0.55  # Extinction coefficient

df["P_gap"] = safe_div(df["Rgrd"], df["Rall"])
df["LAI_BL"] = -np.log(df["P_gap"].replace(0, np.nan)) / k
df["LAI_proxy"] = safe_div(df["veg 1st"], df["veg same"] + df["veg 2nd"])
df["LPI2"] = safe_div(
    df["grd same"] + 0.5 * (df["grd 1st"] + df["grd 2nd"]),
    df["Rall"] + 0.5 * (df["all 1st"] + df["all 2nd"])
)
df["FCI"] = safe_div(df["veg 1st"] + df["veg same"], df["all 1st"] + df["all same"])
df["LCI"] = safe_div(df["veg 2nd"] + df["veg same"], df["all 2nd"] + df["all same"])
df["ABRI"] = safe_div(df["Rveg"], df["Rgrd"])
df["log_ABRI"] = np.log(df["ABRI"].replace(0, np.nan))

print("Computed 8 LiDAR indices: P_gap, LAI_BL, LAI_proxy, LPI2, FCI, LCI, ABRI, log_ABRI")

# Save
df.to_csv(output_file, index=False)
print(f"Saved to: {output_file}")

# Summary
indices = ["P_gap", "LAI_BL", "LAI_proxy", "LPI2", "FCI", "LCI", "ABRI", "log_ABRI"]
print("\nSummary:")
for idx in indices:
    stats = df[idx].describe()
    nan_count = df[idx].isna().sum()
    print(f"{idx:12s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, NaN={nan_count}")
