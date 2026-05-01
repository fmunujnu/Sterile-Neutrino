"""
FDR / Flight Data Recorder – CSV Visualizer (MODIFIED)
=====================================================

改动说明：
✔ 不降采样（适用于稀疏数据）
✔ 每10列一组
✔ 稀疏数据绘图优化（marker + 仅绘制有效点）
✔ 图高度自适应，保证可读性

用法:
    python fdr_plot.py your_data.csv
"""

import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH   = "ExatSample.csv"
DPI        = 150
FIG_WIDTH  = 24
SAVE_PNG   = True
OUTPUT_DIR = "."

GROUP_SIZE = 10   # ⭐ 每组10列（可改5更清晰）

# ─────────────────────────────────────────────────────────────────────────────
# 1. 读取头
# ─────────────────────────────────────────────────────────────────────────────
print(f"[1/5] 读取文件头: {CSV_PATH}")

raw_header = pd.read_csv(
    CSV_PATH, header=None, nrows=14,
    low_memory=False, encoding_errors="replace"
)

names_row = [str(x).strip() for x in raw_header.iloc[11].tolist()]
units_row = [str(x).strip() for x in raw_header.iloc[12].tolist()]
dtype_row = [str(x).strip() for x in raw_header.iloc[13].tolist()]

# ─────────────────────────────────────────────────────────────────────────────
# 2. 读取数据
# ─────────────────────────────────────────────────────────────────────────────
print("[2/5] 读取数据...")

df_raw = pd.read_csv(
    CSV_PATH,
    skiprows=14,
    header=None,
    low_memory=False,
    encoding_errors="replace",
    na_values=["", " ", "nan", "NaN", "N/A"]
)

n_cols = min(len(names_row), df_raw.shape[1])
df_raw = df_raw.iloc[:, :n_cols]
df_raw.columns = names_row[:n_cols]

names_row = names_row[:n_cols]
units_row = (units_row + [""] * n_cols)[:n_cols]
dtype_row = (dtype_row + ["NUMBER"] * n_cols)[:n_cols]

# 转数值
print("[3/5] 转换数值...")
for col in df_raw.columns:
    df_raw[col] = pd.to_numeric(
        df_raw[col].astype(str).str.strip(),
        errors="coerce"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3. 解析元数据（离散）
# ─────────────────────────────────────────────────────────────────────────────
_ENUM_RE = re.compile(r'([\d.eE+\-]+)\s*:\s*[\d.eE+\-]+\s*=\s*"([^"]*)"')

def parse_dtype(dtype_str):
    s = dtype_str.strip()
    if s.upper() == "NUMBER" or not s.startswith("%"):
        return False, {}
    enum_map = {}
    for v, lbl in _ENUM_RE.findall(s):
        try:
            enum_map[float(v)] = lbl
        except:
            pass
    return True, enum_map

meta = {}
for i, name in enumerate(names_row):
    is_disc, emap = parse_dtype(dtype_row[i])
    meta[name] = {
        "unit": units_row[i],
        "is_discrete": is_disc,
        "enum_map": emap,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4. 时间列（不降采样）
# ─────────────────────────────────────────────────────────────────────────────
time_col = names_row[0]
for n in names_row:
    if any(k in n.lower() for k in ("time", "utc", "sec")):
        time_col = n
        break

print(f"    时间列: {time_col}")
print(f"    总行数: {len(df_raw):,}（不降采样）")

t = df_raw[time_col].values.astype(float)
df = df_raw.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. 每10列分组
# ─────────────────────────────────────────────────────────────────────────────
data_cols = [c for c in names_row if c != time_col]

group_cols = {}
for i in range(0, len(data_cols), GROUP_SIZE):
    group_name = f"Group {i//GROUP_SIZE + 1}"
    group_cols[group_name] = data_cols[i:i+GROUP_SIZE]

print(f"\n[4/5] 分组 ({len(group_cols)} 组):")
for g, cols in group_cols.items():
    print(f"    {g}: {len(cols)} 列")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 绘图函数（稀疏优化）
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#e41a1c","#377eb8","#4daf4a","#ff7f00","#984ea3",
    "#a65628","#f781bf","#999999","#1f78b4","#b2df8a"
]

def plot_group(name, cols):
    cont = [c for c in cols if not meta[c]["is_discrete"]]
    disc = [c for c in cols if meta[c]["is_discrete"]]

    cont_h = max(6, len(cols) * 0.6)
    disc_h = 1.0
    fig_h = cont_h + len(disc) * disc_h

    fig = plt.figure(figsize=(FIG_WIDTH, fig_h), dpi=DPI)
    gs = gridspec.GridSpec(1 + len(disc), 1,
                           height_ratios=[cont_h] + [disc_h]*len(disc),
                           hspace=0.08)

    ax = fig.add_subplot(gs[0])

    # 连续数据
    for i, col in enumerate(cont):
        y = df[col].values
        mask = ~np.isnan(y)

        if mask.sum() < 2:
            continue

        ax.plot(
            t[mask], y[mask],
            color=PALETTE[i % len(PALETTE)],
            linewidth=0.6,
            marker='o',
            markersize=2,
            alpha=0.7,
            label=col
        )

    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title(name)

    # 离散数据
    for i, col in enumerate(disc):
        axd = fig.add_subplot(gs[1+i], sharex=ax)
        y = df[col].values
        mask = ~np.isnan(y)

        if mask.sum():
            axd.step(t[mask], y[mask], where="post")

        emap = meta[col]["enum_map"]
        if emap:
            axd.set_yticks(list(emap.keys()))
            axd.set_yticklabels(list(emap.values()), fontsize=6)

        axd.set_ylabel(col, fontsize=6, rotation=0, ha="right")
        axd.grid(True, alpha=0.3)

    ax.set_xlabel("Time")

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 7. 生成图
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] 生成图像...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0
for i, (name, cols) in enumerate(group_cols.items()):
    print(f"  [{i+1}/{len(group_cols)}] {name}")
    fig = plot_group(name, cols)

    path = os.path.join(OUTPUT_DIR, f"FDR_{i+1:02d}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    print(f"    → {path}")

    count += 1

print(f"\n完成！共生成 {count} 张图")
plt.show()