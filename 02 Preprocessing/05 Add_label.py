import numpy as np
import pandas as pd

"""
Script Description:
This script reads the merged smart meter data with survey features from "smart_meter_with_survey_features.csv"
and constructs a target label "future_6h_consumption", which represents the electricity consumption in the next ~6 hours
The output is saved as "05 Smart_meter_with_survey_features_added_label.csv".
"""


# ===== 1. Configuration =====
INPUT_PATH = r"01 Dataset\02 Processed Data\04 Smart_meter_with_survey_features.csv"
OUTPUT_PATH = r"01 Dataset\02 Processed Data\05 Smart_meter_with_survey_features_added_label.csv"

print("Input file:", INPUT_PATH)
print("Output file:", OUTPUT_PATH)

# ===== 2. Read data =====
df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)
print("Column sample:", df.columns[:15])

# ===== 3. Construct timestamp & clean TOTAL_IMPORT (kWh) =====

# DATE + TIME -> timestamp
df["DATE"] = df["DATE"].astype(str).str.strip()
df["TIME"] = df["TIME"].astype(str).str.strip()
df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], errors="coerce")

before = len(df)
df = df.dropna(subset=["timestamp"]).copy()
print("  Dropped rows with unparseable timestamps:", before - len(df))

# Convert TOTAL_IMPORT (kWh) to numeric
col_import = "TOTAL_IMPORT (kWh)"
df[col_import] = df[col_import].astype(str).str.replace(",", "").str.strip()  # Remove thousand separators
df[col_import] = pd.to_numeric(df[col_import], errors="coerce")

before = len(df)
df = df.dropna(subset=[col_import]).copy()
print("  Dropped rows with non-numeric TOTAL_IMPORT:", before - len(df))
print("TOTAL_IMPORT dtype:", df[col_import].dtype)

# ===== 4. Sort by household_ID + timestamp =====
df = df.sort_values(["household_ID", "timestamp"]).reset_index(drop=True)

# ===== 5. Calculate time difference and electricity consumption for the "next ~6 hours" =====

# For each household, get the "next record" timestamp and reading
df["next_timestamp"] = df.groupby("household_ID")["timestamp"].shift(-1)
df["next_import"] = df.groupby("household_ID")[col_import].shift(-1)

# Time difference (in hours)
df["delta_hours_next"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds() / 3600.0

# Set tolerance range for "approximately 6 hours" (you can also change it to 4~8 hours, etc.)
MIN_HOURS = 4
MAX_HOURS = 8

mask_valid_next = df["delta_hours_next"].between(MIN_HOURS, MAX_HOURS)

# Calculate future 6-hour electricity consumption label: next_import - current_import
df["future_6h_consumption"] = np.where(mask_valid_next, df["next_import"] - df[col_import], np.nan)

# Remove non-positive values (to prevent meter rollback or anomalies)
df.loc[df["future_6h_consumption"] <= 0, "future_6h_consumption"] = np.nan

print("Total rows with label (including NaN):", len(df))
print("Rows with valid future 6-hour consumption label:", df["future_6h_consumption"].notna().sum())
df[["household_ID", "timestamp", col_import, "next_timestamp", "delta_hours_next", "future_6h_consumption"]].head()

# ===== 6. Construct modeling table: keep only rows with valid future_6h_consumption =====
df_labelled = df.dropna(subset=["future_6h_consumption"]).reset_index(drop=True)
print("\nShape of df_labelled for modeling:", df_labelled.shape)

# Save results
df_labelled.to_csv(OUTPUT_PATH, index=False)
print("\nSaved data with future_6h_consumption label to:", OUTPUT_PATH)
