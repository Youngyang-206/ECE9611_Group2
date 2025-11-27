import os

import pandas as pd

"""
Script Description:
Since the dataset involved three successive questionnaire surveys, we only selected users who participated in the w1 questionnaire.
This script selects households with wave1 survey dates in January 2024 from "survey_dates.csv".
The output is saved as "02 Valid_w1_households_2024_01.csv".
"""

# ===== 1. Configuration =====
DATA_ROOT = r"01 Dataset\01 Raw Data from IEEE DataPort\02 Survey data"
SURVEY_DATES_PATH = os.path.join(DATA_ROOT, "survey_dates.csv")
OUT_DIR = r"01 Dataset\02 Processed Data"
OUTPUT_PATH = os.path.join(OUT_DIR, "02 Valid_w1_households_2024_01.csv")

print("Input :", SURVEY_DATES_PATH)
print("Output:", OUTPUT_PATH)

# ===== 2. Read data =====
df = pd.read_csv(SURVEY_DATES_PATH, sep=None, engine="python")
df.columns = df.columns.str.strip()
print("Columns:", df.columns.tolist())

# ===== 3. Identify household_ID column (try to detect automatically) =====
hh_col = None
for c in df.columns:
    cl = c.lower().replace(" ", "").replace("-", "").replace("_", "")
    if cl in ["householdid", "hhid", "household_id"]:
        hh_col = c
        break

if hh_col is None:
    # If column name is already 'household_ID', you can directly write:
    # hh_col = "household_ID"
    raise KeyError("Did not find household_ID / HHID column, please check the column names in survey_dates.csv.")

print("Identified household column:", hh_col)
# ===== 4. Parse wave1 date =====
if "wave1" not in df.columns:
    raise KeyError("Did not find column 'wave1', please check if the column name is 'wave1' in the file.")

# wave1 is a date string, e.g., 2024-01-15 or similar
df["wave1_datetime"] = pd.to_datetime(df["wave1"], errors="coerce")

# Drop rows with unparseable dates
before = len(df)
df = df.dropna(subset=["wave1_datetime"]).copy()
print(f"Dropped rows with unparseable wave1 dates: {before - len(df)}")

# Year and month
df["wave1_year"] = df["wave1_datetime"].dt.year
df["wave1_month"] = df["wave1_datetime"].dt.month

# ===== 5. Filter: January 2024 =====
mask_2024_jan = (df["wave1_year"] == 2024) & (df["wave1_month"] == 1)

valid_households = df.loc[mask_2024_jan, hh_col].dropna().drop_duplicates().sort_values().reset_index(drop=True)

# Print total number of users
print("\nNumber of households with wave1 in January 2024:", len(valid_households))

# Preview first few rows
out_df = pd.DataFrame({"household_ID": valid_households})

# Save to CSV (if needed)
out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print("Saved valid household_ID list to:", OUTPUT_PATH)
