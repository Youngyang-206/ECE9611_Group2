# %% [markdown]
# 从 survey_dates.csv 中筛选：wave1 日期在 2024 年 1 月 的 household_ID

import os

import pandas as pd

# ===== 1. 配置路径 =====
DATA_ROOT = r"survey_data"
SURVEY_DATES_PATH = os.path.join(DATA_ROOT, "survey_dates.csv")
OUTPUT_PATH = os.path.join(DATA_ROOT, "valid_w1_households_2024_01.csv")

print("Input :", SURVEY_DATES_PATH)
print("Output:", OUTPUT_PATH)

# ===== 2. 读入数据 =====
df = pd.read_csv(SURVEY_DATES_PATH, sep=None, engine="python")
df.columns = df.columns.str.strip()
print("列名：", df.columns.tolist())

# ===== 3. household_ID 列（尽量自动识别） =====
hh_col = None
for c in df.columns:
    cl = c.lower().replace(" ", "").replace("-", "").replace("_", "")
    if cl in ["householdid", "hhid", "household_id"]:
        hh_col = c
        break

if hh_col is None:
    # 如果你的列名本来就叫 'household_ID'，也可以直接写：
    # hh_col = "household_ID"
    raise KeyError("没有找到 household_ID / HHID 列，请检查 survey_dates.csv 的列名。")

print("识别到 household 列：", hh_col)

# ===== 4. 解析 wave1 日期 =====
if "wave1" not in df.columns:
    raise KeyError("没有找到列名 'wave1'，请确认文件中列名是否为 wave1。")

# wave1 是日期字符串，例如 2024-01-15 之类
df["wave1_datetime"] = pd.to_datetime(df["wave1"], errors="coerce")

# 丢掉无法解析日期的行
before = len(df)
df = df.dropna(subset=["wave1_datetime"]).copy()
print(f"丢弃无法解析 wave1 日期的行数：{before - len(df)}")

# 年和月
df["wave1_year"] = df["wave1_datetime"].dt.year
df["wave1_month"] = df["wave1_datetime"].dt.month

# ===== 5. 过滤：2024 年 1 月 =====
mask_2024_jan = (df["wave1_year"] == 2024) & (df["wave1_month"] == 1)

valid_households = df.loc[mask_2024_jan, hh_col].dropna().drop_duplicates().sort_values().reset_index(drop=True)

# 打印总用户数
print("\n✅ wave1 在 2024 年 1 月的 household 数量:", len(valid_households))

# 预览前几行
out_df = pd.DataFrame({"household_ID": valid_households})

# 保存到 CSV（如果你需要）
out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print("已保存有效 household_ID 列表到：", OUTPUT_PATH)
