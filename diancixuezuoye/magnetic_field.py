import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def process_and_interpolate(file_path, new_x_center):
    """读取单个文件并返回插值后的 y 值和误差"""
    df = pd.read_csv(file_path)
    # 提取原始数据
    x_low = df.iloc[:, 0].values
    x_high = df.iloc[:, 1].values
    y_vals = df.iloc[:, 2].values
    y_errs = df.iloc[:, 3].values
    x_centers = (x_low + x_high) / 2.0

    # 创建插值函数 (使用 quadratic 保证平滑)
    f_y = interp1d(x_centers, y_vals, kind='quadratic', fill_value="extrapolate")
    f_err = interp1d(x_centers, y_errs, kind='linear', fill_value="extrapolate")

    # 计算新点
    new_y = f_y(new_x_center)
    new_err = f_err(new_x_center)

    # 业务逻辑：0-0.1 GeV 设为 0
    new_y[new_x_center < 0.1] = 0.0
    new_err[new_x_center < 0.1] = 0.0

    return new_y, new_err, y_vals[-1], y_errs[-1]

def plot_interpolated_comparison(file_fc, file_pc):
    # 1. 定义新的均匀 Binning (0 to 2.5 GeV, width 0.1)
    bin_width = 0.1
    new_bins_low = np.arange(0, 2.5, bin_width)
    new_bins_high = new_bins_low + bin_width
    new_x_center = (new_bins_low + new_bins_high) / 2.0
    
    # 2. 处理 FC 和 PC
    y_fc_new, err_fc_new, last_y_fc, last_err_fc = process_and_interpolate(file_fc, new_x_center)
    y_pc_new, err_pc_new, last_y_pc, last_err_pc = process_and_interpolate(file_pc, new_x_center)

    # 3. 计算加和 (Total)
    y_total_new = y_fc_new + y_pc_new
    err_total_new = np.sqrt(err_fc_new**2 + err_pc_new**2)
    
    # 处理最后一个 bin (>2.5 GeV)
    last_y_total = last_y_fc + last_y_pc
    last_err_total = np.sqrt(last_err_fc**2 + last_err_pc**2)

    # 整合最终数组 (25个均匀 + 1个最后非均匀)
    final_x = np.append(new_x_center, 2.6) # 2.6 作为最后一部分的示意中心
    final_y_total = np.append(y_total_new, last_y_total)
    final_err_total = np.append(err_total_new, last_err_total)
    
    # 绘图阶梯坐标
    plot_x_steps = np.append(new_bins_low, [2.5, 3.5]) # 示意上限

    # 4. 绘图
    plt.figure(figsize=(10, 7))

    def draw_component(x, y, yerr, x_steps, label, color, is_sum=False):
        # 绘制误差棒
        plt.errorbar(x, y, yerr=yerr, fmt='o', markersize=4,
                     capsize=3, label=label, color=color, zorder=3)
        # 绘制阶梯线
        y_steps = np.append(y, y[-1])
        linewidth = 2.5 if is_sum else 1.5
        plt.step(x_steps, y_steps, where='post', color=color,
                 alpha=0.7, lw=linewidth, zorder=2)

    # 绘制各组分
    draw_component(final_x, np.append(y_fc_new, last_y_fc), 
                   np.append(err_fc_new, last_err_fc), plot_x_steps, "Selection FC (Interp)", "tab:blue")
    
    draw_component(final_x, np.append(y_pc_new, last_y_pc), 
                   np.append(err_pc_new, last_err_pc), plot_x_steps, "Selection PC (Interp)", "tab:orange")

    # 绘制总和
    draw_component(final_x, final_y_total, final_err_total, plot_x_steps, "Total (FC + PC)", "tab:red", is_sum=True)

    # 5. 图表修饰
    plt.xlabel('True Neutrino Energy [GeV]', fontsize=12)
    plt.ylabel('Efficiency', fontsize=12)
    plt.title(r'$\nu_e$ CC Selection Efficiency (Interpolated 0.1 GeV Bins)', fontsize=14)
    plt.axvline(2.5, color='black', linestyle='--', alpha=0.3)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='upper right')
    plt.xlim(0, 3.0)
    plt.ylim(0, min(1.0, max(final_y_total) * 1.3))

    plt.tight_layout()
    plt.show()

    # 打印输出结果
    print("Bin_Low, Bin_High, Total_Eff, Total_Err")
    for i in range(len(new_bins_low)):
        print(f"{new_bins_low[i]:.2f}, {new_bins_high[i]:.2f}, {y_total_new[i]:.6f}, {err_total_new[i]:.6f}")
    print(f"2.50, inf, {last_y_total:.6f}, {last_err_total:.6f}")

# --- 调用执行 ---
# 请确保这两个文件存在于你的目录下
plot_interpolated_comparison("microboone_nu_eCC_FC_Efficiency.csv", "microboone_nu_eCC_PC_Efficiency.csv")