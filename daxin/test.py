import pandas as pd
import numpy as np
from scipy import stats

# =====================================================
# 1. 基础设置与数据读取
# =====================================================
INPUT_FILE = "361778133_按序号_当代大学生网络生活对自我认知的影响_24_24.xlsx"
df = pd.read_excel(INPUT_FILE)
df = df.iloc[:, 6:]
df = df.rename(columns={
    df.columns[0]: "Gender",
    df.columns[1]: "Grade",
    df.columns[2]: "Chat_Time",
    df.columns[3]: "SocialMedia_Time",
    df.columns[4]: "Internet_Time"
})

# =====================================================
# 2. 识别量表题项并清洗数据
# =====================================================
attention_keywords = ["本题请选", "第二个选项", "第五个选项"]
likert_cols = []
for col in df.columns[5:]:
    is_attention = any(key in str(col) for key in attention_keywords)
    if not is_attention:
        likert_cols.append(col)

def is_same_answer(row):
    values = row[likert_cols].dropna().values
    if len(values) == 0:
        return True
    return len(set(values)) == 1

df = df[~df.apply(is_same_answer, axis=1)].copy()

# 计算自我认知总分（直接累加原始分）
df["Likert_Total_Score"] = df[likert_cols].sum(axis=1)

# =====================================================
# 3. 将上网时间选项转化为实际时长 (小时)
# =====================================================
# 时间区间中值映射
time_mapping = {
    1: 0.5,
    2: 2.0,
    3: 4.0,
    4: 6.0
}

df["Chat_Hours"] = df["Chat_Time"].map(time_mapping)
df["Social_Hours"] = df["SocialMedia_Time"].map(time_mapping)
df["Game_Hours"] = df["Internet_Time"].map(time_mapping)

# 计算总用网时长
df["Total_Internet_Hours"] = df["Chat_Hours"] + df["Social_Hours"] + df["Game_Hours"]

# =====================================================
# 4. 相关性分析 (包含总用网时间)
# =====================================================
print("-" * 95)
print("相关性分析 (将用网时间转化为具体小时数，并计算总时长)")
print("-" * 95)

corr_cols = ["Chat_Hours", "Social_Hours", "Game_Hours", "Total_Internet_Hours", "Likert_Total_Score"]
corr_labels = {
    "Chat_Hours": "1.网络聊天(h)",
    "Social_Hours": "2.社交媒体(h)",
    "Game_Hours": "3.网络游戏(h)",
    "Total_Internet_Hours": "4.总用网时间(h)",
    "Likert_Total_Score": "5.自我认知总分"
}

corr_data = df[corr_cols].dropna()
corr_matrix = corr_data.corr(method="pearson")
p_matrix = pd.DataFrame(np.zeros((len(corr_cols), len(corr_cols))), columns=corr_cols, index=corr_cols)

for i in corr_cols:
    for j in corr_cols:
        r, p = stats.pearsonr(corr_data[i], corr_data[j])
        p_matrix.loc[i, j] = p

print(f"{'变量名':<16} | {'1':<10} | {'2':<10} | {'3':<10} | {'4':<10} | {'5':<10}")
print("-" * 95)

for idx, col in enumerate(corr_cols):
    row_str = f"{corr_labels[col]:<14} | "
    for col_other in corr_cols:
        r_val = corr_matrix.loc[col, col_other]
        p_val = p_matrix.loc[col, col_other]
        
        if p_val < 0.01:
            sig_star = "**"
        elif p_val < 0.05:
            sig_star = "*"
        else:
            sig_star = ""
            
        row_str += f"{r_val:.3f}{sig_star:<2}     | "
    print(row_str[:-3])

print("-" * 95)
print("注：* p < 0.05，** p < 0.01（双尾检验）。")