# %% [markdown]
# # Build Wave 1 Household-level Features (one row per household)

import os
from functools import reduce

import numpy as np
import pandas as pd

# ===== 1. 配置：Wave 1 文件所在目录 =====
# 比如：r"D:\rec_dataset\survey_data\wave_1"
W1_DIR = r"survey_data\wave_1"  # TODO: 改成你自己的路径

# 各文件路径
PATH_W1_DEMO = os.path.join(W1_DIR, "w1_demographics.csv")
PATH_W1_APPL = os.path.join(W1_DIR, "w1_appliances.csv")
PATH_W1_AC = os.path.join(W1_DIR, "w1_ac_roster.csv")
PATH_W1_FAN = os.path.join(W1_DIR, "w1_fan_roster.csv")
PATH_W1_LIGHT = os.path.join(W1_DIR, "w1_light_roster.csv")
PATH_W1_ROOM = os.path.join(W1_DIR, "w1_room_roster.csv")
PATH_W1_HHINFO = os.path.join(W1_DIR, "w1_household_information_and_history.csv")
PATH_W1_ENERGY = os.path.join(W1_DIR, "w1_electricity_generation_water_heating_cooking.csv")

OUTPUT_W1_FEATURES = os.path.join(W1_DIR, "w1_household_features_merged.csv")


# ===== 2. 小工具：更稳健的 CSV 读取 =====
def read_csv_flexible(path: str) -> pd.DataFrame:
    print(f"读取 {os.path.basename(path)}")
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df


# ===== 3. 各表构造 household-level 特征 =====


def build_demographics_features(path: str) -> pd.DataFrame:
    """从 w1_demographics 构造人口结构特征"""
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)

    # 家庭成员数量
    feats["w1_hh_member_count"] = g.size()

    # 年龄相关
    if "age" in df.columns:
        feats["w1_hh_avg_age"] = g["age"].mean()
        feats["w1_hh_num_children"] = g.apply(lambda x: (x["age"] < 18).sum())
        feats["w1_hh_num_seniors"] = g.apply(lambda x: (x["age"] >= 65).sum())
    else:
        feats["w1_hh_avg_age"] = np.nan
        feats["w1_hh_num_children"] = np.nan
        feats["w1_hh_num_seniors"] = np.nan

    # 在家时长
    col_home = "no_of_hours_stayed_at_home_during_last_week"
    if col_home in df.columns:
        feats["w1_hh_avg_hours_home"] = g[col_home].mean()

    # 外出工作比例
    col_work = "member_went_out_for_work_or_not_during_last_week"
    if col_work in df.columns:
        feats["w1_hh_share_went_out_for_work"] = g[col_work].apply(
            lambda x: np.mean(x.fillna("").str.startswith("Yes")) if len(x) > 0 else np.nan
        )

    feats = feats.reset_index()
    print("  demographics 特征形状:", feats.shape)
    return feats


def build_appliance_features(path: str) -> pd.DataFrame:
    """从 w1_appliances 构造家电数量 & 使用时长特征"""
    df = read_csv_flexible(path)

    # 每户每种家电使用小时数
    pivot_hours = df.pivot_table(
        index="household_ID",
        columns="appliance_type",
        values="no_of_hours_used_during_last_week",
        aggfunc="sum",
        fill_value=0,
    )
    pivot_hours.columns = [f"w1_appl_hours_{c}" for c in pivot_hours.columns]

    # 每户每种家电数量
    pivot_count = df.pivot_table(
        index="household_ID", columns="appliance_type", values="appliance_ID", aggfunc="count", fill_value=0
    )
    pivot_count.columns = [f"w1_appl_count_{c}" for c in pivot_count.columns]

    feats = pd.concat([pivot_hours, pivot_count], axis=1).reset_index()
    print("  appliances 特征形状:", feats.shape)
    return feats


def build_ac_features(path: str) -> pd.DataFrame:
    """从 w1_ac_roster 构造空调相关特征"""
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)
    feats["w1_num_ac_units"] = g["ac_ID"].nunique()
    feats["w1_ac_total_wattage"] = g["wattage_of_the_ac"].sum()
    feats["w1_ac_hours_day"] = g["no_of_hours_ac_was_on_during_daytime_last_week"].sum()
    feats["w1_ac_hours_night"] = g["no_of_hours_ac_was_on_during_night_last_week"].sum()

    # 变频 AC 比例
    feats["w1_ac_share_inverter"] = g["is_the_ac_inverter_or_not"].apply(
        lambda x: np.mean(x.fillna("").str.strip().str.lower() == "yes")
    )

    feats = feats.reset_index()
    print("  ac 特征形状:", feats.shape)
    return feats


def build_fan_features(path: str) -> pd.DataFrame:
    """从 w1_fan_roster 构造风扇特征"""
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)
    feats["w1_num_fans"] = g["fan_ID"].nunique()
    feats["w1_fan_hours_day"] = g["no_of_hours_fan_was_on_during_daytime_last_week"].sum()
    feats["w1_fan_hours_night"] = g["no_of_hours_fan_was_on_during_night_last_week"].sum()

    feats = feats.reset_index()
    print("  fan 特征形状:", feats.shape)
    return feats


def build_light_features(path: str) -> pd.DataFrame:
    """从 w1_light_roster 构造灯具特征"""
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)
    feats["w1_num_lights"] = g["light_ID"].nunique()
    feats["w1_light_total_wattage"] = g["wattage_of_the_bulb"].sum()
    feats["w1_light_hours_day"] = g["no_of_hours_bulb_was_on_during_daytime_last_week"].sum()
    feats["w1_light_hours_night"] = g["no_of_hours_bulb_was_on_during_night_last_week"].sum()

    feats = feats.reset_index()
    print("  light 特征形状:", feats.shape)
    return feats


def build_room_features(path: str) -> pd.DataFrame:
    """从 w1_room_roster 构造房间结构特征"""
    df = read_csv_flexible(path)
    g = df.groupby("household_ID")

    feats = pd.DataFrame(index=g.size().index)
    feats["w1_num_rooms"] = g["room_ID"].nunique()
    feats["w1_total_windows"] = g["no_of_windows"].sum()
    feats["w1_total_doors_ext"] = g["no_of_doors_opened_to_external_environment"].sum()
    feats["w1_total_room_bulbs"] = g["no_of_bulbs_in_the_room"].sum()
    feats["w1_total_room_fans"] = g["no_of_fans_in_the_room"].sum()
    feats["w1_total_room_acs"] = g["no_of_ACs_in_the_room"].sum()

    # 房间用途里包含 "Bedroom" 的数量
    feats["w1_num_bedrooms"] = g.apply(
        lambda x: np.sum(x["main_purpose_of_the_room"].fillna("").str.contains("Bedroom"))
    )

    feats = feats.reset_index()
    print("  room 特征形状:", feats.shape)
    return feats


def build_household_info_features(path: str) -> pd.DataFrame:
    """从 w1_household_information_and_history 保留关键户级信息"""
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
    print("  household_info 特征形状:", feats.shape)
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
    print("  energy 特征形状:", feats.shape)
    return feats


# ===== 4. 调用所有构造函数并按 household_ID 合并 =====

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


def merge_two(left, right):
    return pd.merge(left, right, on="household_ID", how="outer")


w1_household_features = reduce(merge_two, feature_tables)

print("\n最终 Wave 1 household 特征表形状:", w1_household_features.shape)

# 保存结果
w1_household_features.to_csv(OUTPUT_W1_FEATURES, index=False)
print("已保存到:", OUTPUT_W1_FEATURES)
