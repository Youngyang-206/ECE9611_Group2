import os

import pandas as pd

# ===== 1. 配置路径（根据你的目录结构修改） =====

SMART_CSV_PATH = os.path.join("6hour_interval", "smart_meter_6h_merge.csv")
VALID_HH_PATH = os.path.join("survey_data", "valid_w1_households_2024_01.csv")
FEATURES_CSV_PATH = os.path.join("survey_data", "household_features_merged.csv")

OUTPUT_PATH = os.path.join("6hour_interval", "smart_meter_6h_merge_with_features.csv")

print("smart_meter_6h_merge.csv:", SMART_CSV_PATH)
print("valid_w1_households_2024_01.csv:", VALID_HH_PATH)
print("household_features_merged.csv:", FEATURES_CSV_PATH)
print("输出文件:", OUTPUT_PATH)

# ===== 2. 读入三个文件 =====
smart_df = pd.read_csv(SMART_CSV_PATH)
valid_ids_df = pd.read_csv(VALID_HH_PATH)
features_df = pd.read_csv(FEATURES_CSV_PATH)

smart_df.columns = smart_df.columns.str.strip()
valid_ids_df.columns = valid_ids_df.columns.str.strip()
features_df.columns = features_df.columns.str.strip()

print("smart_df shape:", smart_df.shape)
print("valid_ids_df shape:", valid_ids_df.shape)
print("features_df shape:", features_df.shape)

# ===== 3. household_ID 列名统一检查 =====
# 假设三张表的列名都是 household_ID，如果不是，可以在这里调整
hh_col = "household_ID"
if hh_col not in smart_df.columns:
    raise KeyError(f"{SMART_CSV_PATH} 中没有列 {hh_col}")
if hh_col not in valid_ids_df.columns:
    raise KeyError(f"{VALID_HH_PATH} 中没有列 {hh_col}")
if hh_col not in features_df.columns:
    raise KeyError(f"{FEATURES_CSV_PATH} 中没有列 {hh_col}")

# ===== 4. 过滤 smart_df：只保留 valid household_ID =====
valid_ids = set(valid_ids_df[hh_col].dropna().unique())
print("有效 household_ID 数量:", len(valid_ids))

smart_filtered = smart_df[smart_df[hh_col].isin(valid_ids)].copy()
print("过滤后 smart_filtered shape:", smart_filtered.shape)

# ===== 5. 从 household_features_merged 中选取需要的列 =====
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

# 只保留在 features_df 中实际存在的列，避免 KeyError
available_cols = [c for c in cols_needed if c in features_df.columns]
missing_cols = [c for c in cols_needed if c not in features_df.columns]

print("\n在 household_features_merged.csv 中找到的列数:", len(available_cols))
print("缺失的列数:", len(missing_cols))
if missing_cols:
    print("缺失的列（仅提醒，不影响运行）:")
    for c in missing_cols:
        print("  -", c)

features_sub = features_df[[hh_col] + available_cols].copy()
print("features_sub shape:", features_sub.shape)

# ===== 6. 把特征 merge 到 smart_filtered 上 =====
merged_df = smart_filtered.merge(features_sub, on=hh_col, how="left")
print("合并后 merged_df shape:", merged_df.shape)


# ===== 7. 保存结果 =====
merged_df.to_csv(OUTPUT_PATH, index=False)
print("\n✅ 已保存带有 survey 特征的 smart meter 数据到:", OUTPUT_PATH)
