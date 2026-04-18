import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_covariance_matrix(csv_path, make_symmetric=False):

    col_names = ["bin index column", "bin index row", "covariance"]

    df = pd.read_csv(
        csv_path,
        comment="#",
        names=col_names,
        header=None
    )

    # 删除非数值行（表头、空行等）
    df = df[pd.to_numeric(df["covariance"], errors="coerce").notnull()]

    # 转数值
    df["bin index column"] = pd.to_numeric(df["bin index column"])
    df["bin index row"] = pd.to_numeric(df["bin index row"])
    df["covariance"] = pd.to_numeric(df["covariance"])

    # 自动判断 index 起始
    min_index = int(min(df["bin index column"].min(), df["bin index row"].min()))
    max_index = int(max(df["bin index column"].max(), df["bin index row"].max()))

    if min_index == 0:
        offset = 0
        size = max_index + 1
    elif min_index == 1:
        offset = 1
        size = max_index
    else:
        raise ValueError(f"无法识别的 bin 起始值: min_index={min_index}")

    print(f"Detected bin range: {min_index} → {max_index}, matrix size = {size}")

    cov_matrix = np.zeros((size, size))

    # 填充矩阵
    for _, row in df.iterrows():
        i = int(row["bin index row"]) - offset
        j = int(row["bin index column"]) - offset

        # 越界保护
        if not (0 <= i < size and 0 <= j < size):
            raise ValueError(f"索引越界: i={i}, j={j}, size={size}")

        cov_matrix[i, j] = row["covariance"]

        if make_symmetric:
            cov_matrix[j, i] = cov_matrix[i, j]

    return cov_matrix


def read_three_spectra(csv_path, column_index=3):
    """
    从同一个 HEPData CSV 文件中读取三组谱：
    Data / Background / Signal+Background
    只使用第4列，并根据第4列表头识别块

    返回
    -------
    data : np.ndarray
    bkg : np.ndarray
    sigbkg : np.ndarray
    """

    data = []
    bkg = []
    sigbkg = []

    current_block = None

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # 跳过注释
            if line.startswith("#"):
                continue

            parts = line.split(",")

            # ---------- 表头检测：看第4列 ----------
            if len(parts) > column_index:
                header_name = parts[column_index].strip()

                if header_name == "Data [counts per bin]":
                    current_block = "data"
                    continue

                elif header_name == "Background [counts per bin]":
                    current_block = "bkg"
                    continue

                elif header_name == "Signal + Background [counts per bin]":
                    current_block = "sigbkg"
                    continue

            # ---------- 读取数值 ----------
            try:
                value = float(parts[column_index])
            except (ValueError, IndexError):
                continue

            if current_block == "data":
                data.append(value)
            elif current_block == "bkg":
                bkg.append(value)
            elif current_block == "sigbkg":
                sigbkg.append(value)

    return np.array(data), np.array(bkg), np.array(sigbkg)




# =============================
# 测试模块（调试用，可删除）
# =============================
if __name__ == "__main__":
    test_file = r"E:\sterile neutrino data\HEPData-ins3088922-v1-Unconstrained_14_channels.csv"
    data, bkg, sigbkg = read_three_spectra(test_file)

    print("Data bins:", data.shape)
    print("Background bins:", bkg.shape)
    print("Signal+Background bins:", sigbkg.shape)

    print("Data first 5:", data[:28])
    print("Bkg first 5:", bkg[:28])
    print("Sig+Bkg first 5:", sigbkg[:28])

    data = data[:52]
    bkg = bkg[:52]
    sigbkg = sigbkg[:52]



    # 绘图：观测数据（保留原有绘图功能）
    # ========== 绘制三条曲线 ==========
    x = np.arange(len(data))

    plt.figure(figsize=(12, 5))

    plt.plot(x, data, marker='o', markersize=3, linestyle='-', color='steelblue', label='Data')
    plt.plot(x, bkg, marker='s', markersize=3, linestyle='--', color='tomato', label='Background')
    plt.plot(x, sigbkg, marker='^', markersize=3, linestyle=':', color='seagreen', label='Signal + Background')

    plt.title('Bin Spectra: Data / Background / Signal+Background')
    plt.xlabel('Bin Index')
    plt.ylabel('Counts per Bin')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 读取绝对协方差矩阵（即原始协方差矩阵）
    test_filecov = r"E:\sterile neutrino data\HEPData-ins3088922-v1-14_channel_covariance_matrix.csv"
    C_abs = read_covariance_matrix(
        test_filecov,
        make_symmetric=False          # 如果只给了半个矩阵就设为True
    )

    print("绝对协方差矩阵维度:", C_abs.shape)
    print("前5x5子矩阵:\n", C_abs[:28, :28])

    # ========== 直接使用原始协方差矩阵计算 χ² (δX²) ==========
    # 使用信号+背景作为预期模型（理论预测）
    model = bkg

    # 残差向量（观测与模型的绝对偏差）
    delta = data - model

    # 对绝对协方差矩阵求逆，计算卡方值
    try:
        C_abs_inv = np.linalg.inv(C_abs)
        chi2_abs = delta @ C_abs_inv @ delta
        print(f"\n使用原始（绝对）协方差矩阵计算的 χ² = {chi2_abs:.4f}")
    except np.linalg.LinAlgError:
        print("\n原始协方差矩阵可能奇异，无法求逆。")

    # 输出自由度参考（数据点数，实际拟合时应减去参数个数）
    ndof = len(data)
    print(f"数据点数（自由度粗略参考）: {ndof}")

    # ========== 相对协方差矩阵计算（可选，已注释）==========
    # 如果需要计算相对协方差矩阵，可取消下面的注释
    """
    # 避免除以零
    eps = 1e-10
    model_safe = np.where(model == 0, eps, model)
    denominator = np.outer(model_safe, model_safe)
    C_rel = C_abs / denominator
    print("\n相对协方差矩阵维度:", C_rel.shape)
    print("相对协方差前5x5子矩阵:\n", C_rel[:5, :5])

    delta_rel = delta / model_safe
    try:
        C_rel_inv = np.linalg.inv(C_rel)
        chi2_rel = delta_rel @ C_rel_inv @ delta_rel
        print(f"使用相对协方差矩阵计算的 χ² = {chi2_rel:.4f}")
    except np.linalg.LinAlgError:
        print("相对协方差矩阵可能奇异，无法求逆。")
    """