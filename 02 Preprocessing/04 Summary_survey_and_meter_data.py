import pandas as pd

"""
Script Description:
This script merges the smart meter data from "smart_meter_6h_merge.csv" with selected household features
from "household_features_merged.csv" for households that participated in the wave1 survey in January 2024.
The merged result is saved as "smart_meter_with_survey_features.csv".
"""

# ===== 1. Configuration =====

SMART_CSV_PATH = r"01 Dataset\02 Processed Data\03 Valid_smart_meter_6h_merge.csv"
VALID_HH_PATH = r"01 Dataset\02 Processed Data\02 Valid_w1_households_2024_01.csv"
FEATURES_CSV_PATH = r"01 Dataset\02 Processed Data\01 Survey_household_features_merged.csv"

OUTPUT_PATH = r"01 Dataset\02 Processed Data\04 Smart_meter_with_survey_features.csv"

print("smart_meter_6h_merge.csv:", SMART_CSV_PATH)
print("valid_w1_households_2024_01.csv:", VALID_HH_PATH)
print("household_features_merged.csv:", FEATURES_CSV_PATH)
print("Output file:", OUTPUT_PATH)

# ===== 2. Read the three files =====
smart_df = pd.read_csv(SMART_CSV_PATH)
valid_ids_df = pd.read_csv(VALID_HH_PATH)
features_df = pd.read_csv(FEATURES_CSV_PATH)

smart_df.columns = smart_df.columns.str.strip()
valid_ids_df.columns = valid_ids_df.columns.str.strip()
features_df.columns = features_df.columns.str.strip()

print("smart_df shape:", smart_df.shape)
print("valid_ids_df shape:", valid_ids_df.shape)
print("features_df shape:", features_df.shape)

# ===== 3. household_ID  =====
hh_col = "household_ID"
if hh_col not in smart_df.columns:
    raise KeyError(f"{SMART_CSV_PATH} 中没有列 {hh_col}")
if hh_col not in valid_ids_df.columns:
    raise KeyError(f"{VALID_HH_PATH} 中没有列 {hh_col}")
if hh_col not in features_df.columns:
    raise KeyError(f"{FEATURES_CSV_PATH} 中没有列 {hh_col}")

# ===== 4. Filter smart_df: keep only valid household_ID =====
valid_ids = set(valid_ids_df[hh_col].dropna().unique())
print("Number of valid household_IDs:", len(valid_ids))

smart_filtered = smart_df[smart_df[hh_col].isin(valid_ids)].copy()
print("Shape of smart_filtered after filtering:", smart_filtered.shape)

# ===== 5. Select needed columns from household_features_merged =====
cols_needed = [
    "w1_hh_member_count",
    "w1_hh_avg_age",
    "w1_hh_num_children",
    "w1_hh_num_seniors",
    "w1_hh_avg_hours_home",
    "w1_hh_share_went_out_for_work",
    "w1_num_fans",
    "w1_fan_hours_day",
    "w1_fan_hours_night",
    "w1_num_lights",
    "w1_light_total_wattage",
    "w1_light_hours_day",
    "w1_light_hours_night",
    "w1_num_rooms",
    "w1_total_windows",
    "w1_total_doors_ext",
    "w1_total_room_bulbs",
    "w1_total_room_fans",
    "w1_total_room_acs",
    "w1_num_bedrooms",
    "own_the_house_or_living_on_rent",
    "built_year_of_the_house",
    "type_of_house",
    "floor_area",
    "is_there_business_carried_out_in_the_household",
    "socio_economic_class",
    "total_monthly_expenditure_of_last_month",
    "method_of_receiving_water",
    "water_heating_method_for_bathing",
    "boil_water_before_drinking",
    "no_of_times_food_cooked_last_week",
    "gas_used_for_cooking",
    "electricity_from_national_grid_used_for_cooking",
    "electricity_generated_using_solar_energy_used_for_cooking",
    "firewood_used_for_cooking",
    "kerosene_used_for_cooking",
    "sawdust_or_paddy_husk_used_for_cooking",
    "biogas_used_for_cooking",
    "coconut_shells_or_charcoal_used_for_cooking",
]

# Only keep columns that actually exist in features_df to avoid KeyError
available_cols = [c for c in cols_needed if c in features_df.columns]
missing_cols = [c for c in cols_needed if c not in features_df.columns]

print("\nNumber of columns found in household_features_merged.csv:", len(available_cols))
print("Number of missing columns:", len(missing_cols))
if missing_cols:
    print("Missing columns (for your information, does not affect execution):")
    for c in missing_cols:
        print("  -", c)

features_sub = features_df[[hh_col] + available_cols].copy()
print("features_sub shape:", features_sub.shape)

# ===== 6. Merge features into smart_filtered =====
merged_df = smart_filtered.merge(features_sub, on=hh_col, how="left")
print("Shape of merged_df after merging:", merged_df.shape)


# ===== 7. Save the result =====
merged_df.to_csv(OUTPUT_PATH, index=False)
print("\nSaved smart meter data with survey features to:", OUTPUT_PATH)
