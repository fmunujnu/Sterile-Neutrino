import numpy as np
import matplotlib.pyplot as plt

def plot_interpolated_efficiency():
    # --- 1. 硬编码原始数据 (第一列: 能量 MeV, 第二列: 效率 Efficiency) ---
    raw_data = np.array([
	[0.3818, 0.3367],
	[0.622, 0.6751],
	[0.7546, 0.9309],
	[0.8615, 1.092],
	[0.9833, 1.192],
    [1.122, 1.234],
    [1.282, 1.292],
	[1.463, 1.401],
	[1.735, 1.571],
	[2.619, 1.977]
    ])

    # 转换为 GeV
    x_raw = raw_data[:, 0] / 1000.0
    y_raw = raw_data[:, 1]
    
    # 构建插值序列 (包含起点 0,0)
    x_orig = np.insert(x_raw, 0, 0.0)
    y_orig = np.insert(y_raw, 0, 0.0)
    err_orig = np.insert(y_raw * 0.05, 0, 0.0)

    # --- 2. 定义新的 Bin 结构 (共 26 个数值) ---
    # 常规 Bin: 0.0 到 2.5 GeV (25个 Bin，边界为 26个点)
    regular_edges = np.linspace(0.0, 2.5, 26)
    
    # 结果数组：前 25 个是常规均匀 Bin，第 26 个是 Overflow Bin (>2.5 GeV)
    new_y = np.zeros(26)
    new_err = np.zeros(26)

    # --- 3. 执行分段插值逻辑 ---
    # A. 处理前 25 个常规 Bin (0.0 - 2.5 GeV)
    for i in range(25):
        low = regular_edges[i]
        high = regular_edges[i+1]
        center = (low + high) / 2.0
        # 0-2.5 范围内进行线性插值
        new_y[i] = np.interp(center, x_orig, y_orig)
        new_err[i] = np.interp(center, x_orig, err_orig)

    # B. 处理第 26 个 Bin (Overflow Bin, >2.5 GeV)
    # 取一个代表性的高能中心点进行插值计算（例如 3.0 GeV 附近）
    overflow_center = 3.0 
    new_y[25] = np.interp(overflow_center, x_orig, y_orig)
    new_err[25] = np.interp(overflow_center, x_orig, err_orig)

    # --- 4. 打印输出数组 ---
    print("Re-binned Efficiency Array (Total 26 bins):")
    print("Indices 0-24: 0.0-2.5 GeV (step 0.1)")
    print("Index 25: Overflow (>2.5 GeV)")
    print(new_y)

    # --- 5. 绘图设置 ---
    plt.figure(figsize=(10, 7))
    
    # 为了绘图显示，定义 26 个 bin 的显示边界
    # 前 25 个 bin 均匀，最后一个显示得宽一点（从 2.5 到 3.5）
    display_low = np.append(regular_edges[:-1], 2.5)
    display_high = np.append(regular_edges[1:], 3.5)
    
    x_centers = (display_low + display_high) / 2.0
    x_errs = (display_high - display_low) / 2.0
    
    x_steps = np.append(display_low, display_high[-1])
    y_steps = np.append(new_y, new_y[-1])

    color = "tab:green"
    label = "Linear Interpolated Efficiency (26 Bins)"
    
    plt.errorbar(x_centers, new_y, xerr=x_errs, yerr=new_err, fmt='o', 
                 markersize=4, capsize=3, label=label, color=color, zorder=3)
    
    plt.step(x_steps, y_steps, where='post', color=color, alpha=0.7, lw=2.5, zorder=2)
    plt.scatter(x_orig, y_orig, color='black', marker='x', label='Original Data', zorder=4)

    plt.axvline(2.5, color='red', linestyle='--', alpha=0.4, label='Split Boundary (2.5 GeV)')
    plt.xlabel('True Neutrino Energy [GeV]', fontsize=12)
    plt.ylabel('Efficiency', fontsize=12)
    plt.title(r'$\nu_e$ CC Efficiency Re-binning (25 Regular + 1 Overflow)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, loc='upper right')
    plt.xlim(0, 3.6)
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_interpolated_efficiency()