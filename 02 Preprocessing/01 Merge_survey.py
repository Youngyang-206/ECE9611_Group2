import os
from functools import reduce

import numpy as np
import pandas as pd

"""
Script Description:
This script merges the survey questionnaire data tables into a single household-level feature table.
The original survey form contains multiple tables, each representing different aspects of household information:
    1) w1_ac_roster.csv: Air conditioner details
    2) w1_appliances.csv: Appliance usage details
    3) w1_demographics.csv: Demographic information of household members
    4) w1_electricity_generation_water_heating_cooking.csv: Energy generation and cooking methods
    5) w1_fan_roster.csv: Fan details
    6) w1_household_information_and_history.csv: General household information
    7) w1_light_roster.csv: Lighting details
    8) w1_room_roster.csv: Room details
The script processes each table to extract relevant household-level features and merges them into a single DataFrame.
"""


# ===== 1. Utility: More robust CSV reading =====
def read_csv_flexible(path: str) -> pd.DataFrame:
    print(f"Reading {os.path.basename(path)}")
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df


# ===== 2. Build household-level features from each table =====
def build_demographics_features(path: str) -> pd.DataFrame:
    """
    Build demographic features from w1_demographics
    Features include:
    - Household member count
    - Average age
    - Number of children (<18)
    - Number of seniors (65+)
    - Average hours stayed at home last week
    - Share of members who went out for work last week

    Parameters:
    path (str): Path to the w1_demographics CSV file.

    Returns:
    pd.DataFrame: DataFrame with household-level demographic features.
    """
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)

    # Household member count
    feats["w1_hh_member_count"] = g.size()

    # Age-related features
    if "age" in df.columns:
        feats["w1_hh_avg_age"] = g["age"].mean()
        feats["w1_hh_num_children"] = g.apply(lambda x: (x["age"] < 18).sum())
        feats["w1_hh_num_seniors"] = g.apply(lambda x: (x["age"] >= 65).sum())
    else:
        feats["w1_hh_avg_age"] = np.nan
        feats["w1_hh_num_children"] = np.nan
        feats["w1_hh_num_seniors"] = np.nan

    # Average hours stayed at home last week
    col_home = "no_of_hours_stayed_at_home_during_last_week"
    if col_home in df.columns:
        feats["w1_hh_avg_hours_home"] = g[col_home].mean()

    # Share of members who went out for work last week
    col_work = "member_went_out_for_work_or_not_during_last_week"
    if col_work in df.columns:
        feats["w1_hh_share_went_out_for_work"] = g[col_work].apply(
            lambda x: np.mean(x.fillna("").str.startswith("Yes")) if len(x) > 0 else np.nan
        )

    feats = feats.reset_index()
    print("demographic feature shape:", feats.shape)
    return feats


def build_appliance_features(path: str) -> pd.DataFrame:
    """
    Build appliance usage features from w1_appliances
    Features include:
    - Total usage hours per appliance type
    - Count of each appliance type

    Parameters:
    path (str): Path to the w1_appliances CSV file.

    Returns:
    pd.DataFrame: DataFrame with household-level appliance features.
    """
    df = read_csv_flexible(path)

    # Total usage hours per appliance type
    pivot_hours = df.pivot_table(
        index="household_ID",
        columns="appliance_type",
        values="no_of_hours_used_during_last_week",
        aggfunc="sum",
        fill_value=0,
    )
    pivot_hours.columns = [f"w1_appl_hours_{c}" for c in pivot_hours.columns]

    # Count of each appliance type
    pivot_count = df.pivot_table(
        index="household_ID", columns="appliance_type", values="appliance_ID", aggfunc="count", fill_value=0
    )
    pivot_count.columns = [f"w1_appl_count_{c}" for c in pivot_count.columns]

    feats = pd.concat([pivot_hours, pivot_count], axis=1).reset_index()
    print("appliances feature shape:", feats.shape)
    return feats


def build_ac_features(path: str) -> pd.DataFrame:
    """
    Build air conditioner features from w1_ac_roster
    Features include:
    - Number of AC units
    - Total wattage
    - Total hours used during daytime and nighttime
    - Share of inverter ACs

    Parameters:
    path (str): Path to the w1_ac_roster CSV file.
    Returns:
    pd.DataFrame: DataFrame with household-level AC features.
    """
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)
    feats["w1_num_ac_units"] = g["ac_ID"].nunique()
    feats["w1_ac_total_wattage"] = g["wattage_of_the_ac"].sum()
    feats["w1_ac_hours_day"] = g["no_of_hours_ac_was_on_during_daytime_last_week"].sum()
    feats["w1_ac_hours_night"] = g["no_of_hours_ac_was_on_during_night_last_week"].sum()
    feats["w1_ac_share_inverter"] = g["is_the_ac_inverter_or_not"].apply(
        lambda x: np.mean(x.fillna("").str.strip().str.lower() == "yes")
    )
    feats = feats.reset_index()
    print("  ac feature shape:", feats.shape)
    return feats


def build_fan_features(path: str) -> pd.DataFrame:
    """
    Build fan features from w1_fan_roster
    Features include:
    - Number of fans
    - Total hours used during daytime and nighttime

    Parameters:
    path (str): Path to the w1_fan_roster CSV file.

    Returns:
    pd.DataFrame: DataFrame with household-level fan features.
    """
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)
    feats["w1_num_fans"] = g["fan_ID"].nunique()
    feats["w1_fan_hours_day"] = g["no_of_hours_fan_was_on_during_daytime_last_week"].sum()
    feats["w1_fan_hours_night"] = g["no_of_hours_fan_was_on_during_night_last_week"].sum()

    feats = feats.reset_index()
    print("fan feature shape:", feats.shape)
    return feats


def build_light_features(path: str) -> pd.DataFrame:
    """
    Build lighting features from w1_light_roster
    Features include:
    - Number of lights
    - Total wattage
    - Total hours used during daytime and nighttime

    Parameters:
    path (str): Path to the w1_light_roster CSV file.

    Returns:
    pd.DataFrame: DataFrame with household-level lighting features.
    """
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)
    feats["w1_num_lights"] = g["light_ID"].nunique()
    feats["w1_light_total_wattage"] = g["wattage_of_the_bulb"].sum()
    feats["w1_light_hours_day"] = g["no_of_hours_bulb_was_on_during_daytime_last_week"].sum()
    feats["w1_light_hours_night"] = g["no_of_hours_bulb_was_on_during_night_last_week"].sum()

    feats = feats.reset_index()
    print("light feature shape:", feats.shape)
    return feats


def build_room_features(path: str) -> pd.DataFrame:
    """
    Build room features from w1_room_roster
    Features include:
    - Number of rooms
    - Total number of windows, doors, bulbs, fans, ACs
    - Number of bedrooms

    Parameters:
    path (str): Path to the w1_room_roster CSV file.

    Returns:
    pd.DataFrame: DataFrame with household-level room features.
    """
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)
    feats["w1_num_rooms"] = g["room_ID"].nunique()
    feats["w1_total_windows"] = g["no_of_windows"].sum()
    feats["w1_total_doors_ext"] = g["no_of_doors_opened_to_external_environment"].sum()
    feats["w1_total_room_bulbs"] = g["no_of_bulbs_in_the_room"].sum()
    feats["w1_total_room_fans"] = g["no_of_fans_in_the_room"].sum()
    feats["w1_total_room_acs"] = g["no_of_ACs_in_the_room"].sum()
    feats["w1_num_bedrooms"] = g.apply(
        lambda x: np.sum(x["main_purpose_of_the_room"].fillna("").str.contains("Bedroom"))
    )
    feats = feats.reset_index()
    print("room feature shape:", feats.shape)
    return feats


def build_household_info_features(path: str) -> pd.DataFrame:
    """
    Build household information features from w1_household_information_and_history
    Only keep key columns.

    Parameters:
    path (str): Path to the w1_household_information_and_history CSV file.

    Returns:
    pd.DataFrame: DataFrame with household-level household information features.
    """
    df = read_csv_flexible(path)

    keep_cols = [
        "household_ID",
        "no_of_electricity_meters",
        "own_the_house_or_living_on_rent",
        "built_year_of_the_house",
        "type_of_house",
        "no_of_storeys",
        "floor_area",
        "no_of_household_members",
        "is_there_business_carried_out_in_the_household",
        "socio_economic_class",
        "total_monthly_expenditure_of_last_month",
        "type_of_electricity_meter",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    feats = df[keep_cols].drop_duplicates(subset=["household_ID"])
    print("  household_info feature shape:", feats.shape)
    return feats


def build_energy_features(path: str) -> pd.DataFrame:
    """从 w1_electricity_generation_water_heating_cooking 保留关键行为/能源特征"""
    df = read_csv_flexible(path)

    key_cols = [
        "household_ID",
        "have_backup_generator",
        "generate_electicity_using_solar_energy",
        "solar_energy_used_for_water_heating",
        "solar_energy_used_for_cooking",
        "method_of_receiving_water",
        "water_heating_method_for_bathing",
        "boil_water_before_drinking",
        "source_of_energy_for_boiling_drinking_water",
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
    key_cols = [c for c in key_cols if c in df.columns]

    feats = df[key_cols].drop_duplicates(subset=["household_ID"])
    print("  energy feature shape:", feats.shape)
    return feats


def merge_two(left, right):
    """
    Merges two DataFrames on 'household_ID' using an outer join.

    Parameters:
    left (pd.DataFrame): The left DataFrame to merge.
    right (pd.DataFrame): The right DataFrame to merge.

    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    return pd.merge(left, right, on="household_ID", how="outer")


# ===== 3. Main: Build and merge all household-level features =====


W1_DIR = r"01 Dataset\01 Raw Data from IEEE DataPort\02 Survey data"
Out_DIR = r"01 Dataset\02 Processed Data"

# table paths
PATH_W1_DEMO = os.path.join(W1_DIR, "w1_demographics.csv")
PATH_W1_APPL = os.path.join(W1_DIR, "w1_appliances.csv")
PATH_W1_AC = os.path.join(W1_DIR, "w1_ac_roster.csv")
PATH_W1_FAN = os.path.join(W1_DIR, "w1_fan_roster.csv")
PATH_W1_LIGHT = os.path.join(W1_DIR, "w1_light_roster.csv")
PATH_W1_ROOM = os.path.join(W1_DIR, "w1_room_roster.csv")
PATH_W1_HHINFO = os.path.join(W1_DIR, "w1_household_information_and_history.csv")
PATH_W1_ENERGY = os.path.join(W1_DIR, "w1_electricity_generation_water_heating_cooking.csv")

OUTPUT_W1_FEATURES = os.path.join(Out_DIR, "01 Survey_household_features_merged.csv")

demo_feats = build_demographics_features(PATH_W1_DEMO)
appl_feats = build_appliance_features(PATH_W1_APPL)
ac_feats = build_ac_features(PATH_W1_AC)
fan_feats = build_fan_features(PATH_W1_FAN)
light_feats = build_light_features(PATH_W1_LIGHT)
room_feats = build_room_features(PATH_W1_ROOM)
hhinfo_feats = build_household_info_features(PATH_W1_HHINFO)
energy_feats = build_energy_features(PATH_W1_ENERGY)

feature_tables = [
    demo_feats,
    appl_feats,
    ac_feats,
    fan_feats,
    light_feats,
    room_feats,
    hhinfo_feats,
    energy_feats,
]

w1_household_features = reduce(merge_two, feature_tables)

print("\nFinal survey household features shape:", w1_household_features.shape)

w1_household_features.to_csv(OUTPUT_W1_FEATURES, index=False)
print("Saved to", OUTPUT_W1_FEATURES)
