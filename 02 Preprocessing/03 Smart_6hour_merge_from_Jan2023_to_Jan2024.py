# %% [markdown]
# 从 6hour_interval 目录中读取 smart_6hour_1~5.csv
# 筛选 DATE 在 2023-01-01 到 2024-01-31 之间的数据
# 只保留 household_ID, DATE, TIME, TOTAL_IMPORT (kWh)
# 保存为 smart_meter_6h_merge.csv

import glob
import os

import pandas as pd

# ===== 1. 配置路径（改成你自己的项目路径） =====
INTERVAL_DIR = "6hour_interval"

# 输出文件
OUTPUT_PATH = os.path.join(INTERVAL_DIR, "smart_meter_6h_merge.csv")

print("读取目录:", INTERVAL_DIR)
print("输出文件:", OUTPUT_PATH)

# ===== 2. 收集 smart_6hour_*.csv 文件 =====
pattern = os.path.join(INTERVAL_DIR, "smart_6hour_*.csv")
file_list = sorted(glob.glob(pattern))

print("找到文件:")
for f in file_list:
    print("  -", os.path.basename(f))

if not file_list:
    raise FileNotFoundError("没有找到 smart_6hour_*.csv 文件，请检查目录和文件名。")

# ===== 3. 逐个读取并筛选日期范围 =====
dfs = []

# 日期范围：2023-01-01 ~ 2024-01-31
start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp("2024-01-31")

usecols = ["household_ID", "DATE", "TIME", "TOTAL_IMPORT (kWh)"]

for f in file_list:
    print(f"\n读取 {os.path.basename(f)} ...")
    # sep=None + engine="python" 可以自动识别逗号/Tab 等分隔
    df = pd.read_csv(f, sep=None, engine="python", usecols=usecols)
    df.columns = df.columns.str.strip()

    # 解析 DATE 列为日期
    df["DATE"] = pd.to_datetime(df["DATE"].astype(str).str.strip(), errors="coerce")

    # 丢掉解析失败的日期
    before = len(df)
    df = df.dropna(subset=["DATE"])
    print("  丢弃日期解析失败行数:", before - len(df))

    # 筛选 2023-01-01 ~ 2024-01-31 的行
    mask = (df["DATE"] >= start_date) & (df["DATE"] <= end_date)
    df = df.loc[mask, usecols]  # 再次只保留这 4 列，防止混入多余列

    print("  保留行数:", len(df))
    dfs.append(df)

# ===== 4. 合并所有筛选后的数据 =====
merged = pd.concat(dfs, ignore_index=True)
print("\n合并后行数:", len(merged))
print(merged.head())

# 如果需要 DATE 回写成字符串，可以加这一行（可选）：
# merged["DATE"] = merged["DATE"].dt.strftime("%Y-%m-%d")

# ===== 5. 保存到 CSV =====
merged.to_csv(OUTPUT_PATH, index=False)
print("\n✅ 已保存到:", OUTPUT_PATH)
