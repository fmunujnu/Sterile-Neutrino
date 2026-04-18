import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_with_sum(file_fc, file_pc):
    plt.figure(figsize=(10, 7))

    # 1. 读取数据
    df_fc = pd.read_csv(file_fc)
    df_pc = pd.read_csv(file_pc)

    # 提取公共部分 (假设两者 Binning 一致)
    x_low = df_fc.iloc[:, 0]
    x_high = df_fc.iloc[:, 1]
    x_center = (x_low + x_high) / 2
    x_err = (x_high - x_low) / 2
    x_steps = np.append(x_low.values, x_high.values[-1])

    # 提取 y 值和误差
    y_fc = df_fc.iloc[:, 2]
    err_fc = df_fc.iloc[:, 3]

    y_pc = df_pc.iloc[:, 2]
    err_pc = df_pc.iloc[:, 3]

    # 2. 计算加和结果
    y_sum = y_fc + y_pc
    # 统计误差按平方和根计算
    err_sum = np.sqrt(err_fc ** 2 + err_pc ** 2)

    # 3. 绘图函数：封装重复的绘图动作
    def draw_component(x, y, yerr, xerr, label, color, is_sum=False):
        # 绘制误差棒
        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', markersize=4,
                     capsize=3, label=label, color=color, zorder=3)
        # 绘制阶梯线
        y_steps = np.append(y.values, y.values[-1])
        linewidth = 2.5 if is_sum else 1.5
        plt.step(x_steps, y_steps, where='post', color=color,
                 alpha=0.7, lw=linewidth, zorder=2)

    # 绘制 FC
    draw_component(x_center, y_fc, err_fc, x_err, "Selection FC", "tab:blue")

    # 绘制 PC
    draw_component(x_center, y_pc, err_pc, x_err, "Selection PC", "tab:orange")

    # 绘制 Sum (加和)
    draw_component(x_center, y_sum, err_sum, x_err, "Total (FC + PC)", "tab:red", is_sum=True)

    # 4. 图表修饰
    plt.xlabel('True Neutrino Energy [GeV]', fontsize=12)
    plt.ylabel('Efficiency', fontsize=12)
    plt.title(r'$\nu_e$ CC Selection Efficiency Comparison', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='upper right')
    plt.ylim(0, min(1.0, max(y_sum) * 1.2))  # 自动调整高度，上限不超过 1.0

    plt.tight_layout()
    plt.savefig("efficiency_total.png", dpi=300)
    plt.show()


# --- 运行 ---
# 确保你的两个文件名正确
plot_with_sum("microboone_nu_eCC_FC_Efficiency.csv", "microboone_nu_eCC_PC_Efficiency.csv")