import glob
import os

import pandas as pd

"""
Script Description:
This script merges all "smart_6hour_*.csv" files from the "6hour_interval" directory,
filters the data to include only rows with DATE between 2023-01-01 and 2024-01-31,
and retains only the columns "household_ID", "DATE", "TIME", and "TOTAL_IMPORT (kWh)".
The merged result is saved as "smart_meter_6h_merge.csv" in the same directory.
"""

# ===== 1. Configuration =====
INTERVAL_DIR = r"01 Dataset\01 Raw Data from IEEE DataPort\01 Meter data"
OUT_DIR = r"01 Dataset\02 Processed Data"

# Output file
OUTPUT_PATH = os.path.join(OUT_DIR, "03 Valid_smart_meter_6h_merge.csv")

print("Input directory:", INTERVAL_DIR)
print("Output file:", OUTPUT_PATH)
# ===== 2. Collect smart_6hour_*.csv files =====
pattern = os.path.join(INTERVAL_DIR, "smart_6hour_*.csv")
file_list = sorted(glob.glob(pattern))

print("Found files:")
for f in file_list:
    print("  -", os.path.basename(f))

if not file_list:
    raise FileNotFoundError("No smart_6hour_*.csv files found, please check the directory and filenames.")

# ===== 3. Read each file and filter by date range =====
dfs = []

# Date range: 2023-01-01 ~ 2024-01-31
start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp("2024-01-31")

usecols = ["household_ID", "DATE", "TIME", "TOTAL_IMPORT (kWh)"]

for f in file_list:
    print(f"\nReading {os.path.basename(f)} ...")
    df = pd.read_csv(f, sep=None, engine="python", usecols=usecols)
    df.columns = df.columns.str.strip()

    # Parse DATE column as datetime
    df["DATE"] = pd.to_datetime(df["DATE"].astype(str).str.strip(), errors="coerce")

    # Drop rows with unparseable dates
    before = len(df)
    df = df.dropna(subset=["DATE"])
    print("  Dropped rows with unparseable dates:", before - len(df))

    # Filter rows between 2023-01-01 and 2024-01-31
    mask = (df["DATE"] >= start_date) & (df["DATE"] <= end_date)
    df = df.loc[mask, usecols]  # Keep only these 4 columns again to prevent extra columns

    print("  Rows retained:", len(df))
    dfs.append(df)

# ===== 4. Merge all filtered data =====
merged = pd.concat(dfs, ignore_index=True)
print("\nRows after merge:", len(merged))
print(merged.head())


# ===== 5. Save to CSV =====
merged.to_csv(OUTPUT_PATH, index=False)
print("\n Saved to:", OUTPUT_PATH)
