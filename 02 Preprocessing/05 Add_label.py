# %% [markdown]
# 从 smart_meter_6h_merge_with_features.csv 中：
#  - 构造 timestamp
#  - 对每个 household 计算未来 ~6 小时用电量作为标签 future_6h_consumption

import os

import numpy as np
import pandas as pd

# ===== 1. 路径配置 =====
INPUT_PATH = os.path.join("6hour_interval", "smart_meter_6h_merge_with_features.csv")
OUTPUT_PATH = os.path.join("6hour_interval", "smart_meter_6h_with_future6h_label.csv")

print("输入文件:", INPUT_PATH)
print("输出文件:", OUTPUT_PATH)

# ===== 2. 读入数据 =====
df = pd.read_csv(INPUT_PATH)
df.columns = df.columns.str.strip()

print("原始形状:", df.shape)
print("列示例:", df.columns[:15])

# ===== 3. 构造 timestamp & 清洗 TOTAL_IMPORT (kWh) =====

# DATE + TIME -> timestamp
df["DATE"] = df["DATE"].astype(str).str.strip()
df["TIME"] = df["TIME"].astype(str).str.strip()
df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], errors="coerce")

before = len(df)
df = df.dropna(subset=["timestamp"]).copy()
print("  丢弃无法解析时间的行数:", before - len(df))

# 把 TOTAL_IMPORT (kWh) 转成数值
col_import = "TOTAL_IMPORT (kWh)"
df[col_import] = df[col_import].astype(str).str.replace(",", "").str.strip()  # 防止有千分位逗号
df[col_import] = pd.to_numeric(df[col_import], errors="coerce")

before = len(df)
df = df.dropna(subset=[col_import]).copy()
print("  丢弃 TOTAL_IMPORT 无法转数字的行数:", before - len(df))
print("TOTAL_IMPORT dtype:", df[col_import].dtype)

# ===== 4. 按 household_ID + timestamp 排序 =====
df = df.sort_values(["household_ID", "timestamp"]).reset_index(drop=True)

# ===== 5. 计算“未来 ~6 小时”的时间差和用电量 =====

# 对每个 household，拿“下一条记录”的时间和读数
df["next_timestamp"] = df.groupby("household_ID")["timestamp"].shift(-1)
df["next_import"] = df.groupby("household_ID")[col_import].shift(-1)

# 时间差（单位小时）
df["delta_hours_next"] = (df["next_timestamp"] - df["timestamp"]).dt.total_seconds() / 3600.0

# 设置“约等于 6 小时”的容忍范围（你也可以改成 4~8 小时等）
MIN_HOURS = 4
MAX_HOURS = 8

mask_valid_next = df["delta_hours_next"].between(MIN_HOURS, MAX_HOURS)

# 计算未来 6 小时用电量标签：next_import - current_import
df["future_6h_consumption"] = np.where(mask_valid_next, df["next_import"] - df[col_import], np.nan)

# 去掉非正值（防止表计回卷或异常）
df.loc[df["future_6h_consumption"] <= 0, "future_6h_consumption"] = np.nan

print("带标签的总行数（含 NaN）:", len(df))
print("有效未来6小时标签的行数:", df["future_6h_consumption"].notna().sum())

df[["household_ID", "timestamp", col_import, "next_timestamp", "delta_hours_next", "future_6h_consumption"]].head()

# ===== 6. 构造建模用表：只保留有有效 future_6h_consumption 的行 =====
df_labelled = df.dropna(subset=["future_6h_consumption"]).reset_index(drop=True)
print("\n用于建模的 df_labelled 形状:", df_labelled.shape)

# 保存结果
df_labelled.to_csv(OUTPUT_PATH, index=False)
print("\n✅ 已保存带 future_6h_consumption 标签的数据到:", OUTPUT_PATH)
