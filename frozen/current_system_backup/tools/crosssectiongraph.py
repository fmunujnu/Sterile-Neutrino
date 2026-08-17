import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =====================================================================
# 1. 原始提取的数据点 (多行文本格式，无 Python 括号，方便直接粘贴和修改)
# =====================================================================
raw_data_str = """
8.060602615414018, 0.2927032331072895
13.292334844549023, 0.3047424152942002
16.15242761775714, 0.3047424152942002
17.180641336126044, 0.3541003198383038
18.353620664107673, 0.31797055773943994
22.4157132536856, 0.32250332273404003
25.4025126487636, 0.328373606683579
29.362283710970047, 0.3325863621575124
0.7003512435742756, 0.2455233941746389
3.6300520646997056, 0.27518736070544675
2.911067579625909, 0.26745121922343185
2.4793367168681413, 0.331225556937925
4.52661356426634, 0.31735693098115414
3.4179764271202613, 0.3219140545333009
5.994842503189409, 0.30843530713885803
6.761847306683158, 0.30843530713885803
7.939298578908191, 0.3219140545333009
9.703493171471163, 0.31735693098115414
10.51444835982241, 0.2955209235202888
10.305567101753187, 0.3219140545333009
11.859710123376683, 0.3506643701286162
13.377090209314535, 0.3265366166281175
15.088610143633879, 0.3265366166281175
16.349618572738972, 0.3265366166281175
4.0131541219153215, 0.3506643701286162
"""

# 解析多行文本
raw_lines = [line.strip().split(',') for line in raw_data_str.strip().split('\n') if line.strip()]
raw_data = np.array(raw_lines, dtype=float)

x_raw = raw_data[:, 0]
y_raw = raw_data[:, 1]

# =====================================================================
# 2. 先验物理约束与非线性模型定义
# =====================================================================
# X_INTERCEPT: 强约束的 X 轴截距
X_INTERCEPT = 0.11

def physical_model(x, y_max, L, A, B):
    x_arr = np.atleast_1d(x)
    result = np.zeros_like(x_arr, dtype=float)
    
    mask = x_arr > X_INTERCEPT
    dx = x_arr[mask] - X_INTERCEPT
    
    # 饱和项
    base_saturate = y_max * dx / (dx + L)
    # 峰值与下降修正
    correction = 1.0 + A * np.exp(-B * dx)
    
    result[mask] = base_saturate * correction
    return np.clip(result, a_min=0.0, a_max=None)

# =====================================================================
# 3. 非线性最小二乘法拟合 (Robust Curve Fitting)
# =====================================================================
valid_mask = x_raw > X_INTERCEPT
x_filtered = x_raw[valid_mask]
y_filtered = y_raw[valid_mask]

# 初始猜测
p0 = [0.677, 0.5, 0.5, 0.5]

# 边界参数限制
bounds = (
    [0.3, 0.01, -0.9, 0.01],  # 下界
    [2.0, 5.0,  5.0,  1.0]    # 上界
)

popt, pcov = curve_fit(physical_model, x_filtered, y_filtered, p0=p0, bounds=bounds, maxfev=10000)
y_max_fit, L_fit, A_fit, B_fit = popt

def smooth_curve_func(x):
    return physical_model(x, y_max_fit, L_fit, A_fit, B_fit)

# =====================================================================
# 4. 生成 0 到 6 GeV 之间、宽度为 0.1 GeV 的 60 个 Bin 代表值 (Bin Centers)
# =====================================================================
# 定义 Bin 边界：[0.0, 0.1, ..., 6.0]
bin_edges = np.arange(0.0, 6.1, 0.1)

# 定义 Bin 中心点：[0.05, 0.15, ..., 5.95]
bin_centers = bin_edges[:-1] + 0.05

# 计算 60 个 Bin 中心点对应的 Y 轴拟合数据
y_binned_steps = smooth_curve_func(bin_centers)

# =====================================================================
# 5. 终端格式化输出 (精准对应 60 个 Bin 阶梯值，方便直接复制)
# =====================================================================
print("\n" + "="*80)
print(f" 阶梯图对应的 Binned Array 数组 (0.0 至 6.0 GeV, 共 60 个 Bin, 宽度 0.1 GeV):")
print("="*80)
y_array_str = ", ".join([f"{val:.6f}" for val in y_binned_steps])
print(f"binned_array = np.array([\n    {y_array_str}\n])")
print("="*80)
print(" 每一个元素分别对应的物理区间:")
print(" binned_array[0]  -> [0.0, 0.1) GeV (中心点 0.05 GeV)")
print(" binned_array[1]  -> [0.1, 0.2) GeV (中心点 0.15 GeV)")
print(" ...")
print(" binned_array[59] -> [5.9, 6.0) GeV (中心点 5.95 GeV)")
print("="*80 + "\n")

# =====================================================================
# 6. 画图展示 (包含原始数据、强约束平滑曲线、以及与输出数组完全一致的阶梯图)
# =====================================================================
x_dense = np.linspace(0.0, 10.0, 1000)
y_dense = smooth_curve_func(x_dense)

plt.figure(figsize=(10, 6.5))

# A. 绘制原始提取散点图
plt.scatter(x_raw, y_raw, color='black', marker='o', s=35, alpha=0.7, label='Raw Data')

# B. 绘制连续平滑拟合曲线
plt.plot(x_dense, y_dense, color='red', linestyle='-', linewidth=2.2, 
         label=f'Physical Saturation Fit (Intercept={X_INTERCEPT} GeV)\nAsymptotic Limit $\\approx$ {y_max_fit:.3f}')

# C. 绘制精准对应的步长阶梯图 (Step Plot)
# 在 y_binned_steps 尾部追加最后一个元素以实现 61 个边界的完美对齐
y_plot_steps = np.append(y_binned_steps, y_binned_steps[-1])
plt.step(bin_edges, y_plot_steps, where='post', color='blue', linestyle='--', alpha=0.8, linewidth=1.8, 
         label='Binned Step Plot (0-6 GeV, 60 bins)')

# 专门标注先验截距点
plt.axvline(x=X_INTERCEPT, color='green', linestyle=':', alpha=0.7, label=f'Prior Intercept Boundary ({X_INTERCEPT} GeV)')

# 设置坐标轴与图表美化
plt.xlim(-0.2, 10.0)  # 显示至 10 GeV 以看清趋势
plt.ylim(0.0, 1.5)
plt.xlabel('$E_\\nu$ (GeV) [Linear Scale]', fontsize=12)
plt.ylabel('$\\sigma_{CC} / E_\\nu$ ($10^{-38}$ cm$^2$ / GeV)', fontsize=12)
plt.title(f'Physical Saturation Model Fitting & Binning (Intercept = {X_INTERCEPT} GeV)', fontsize=14)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(fontsize=10, loc='upper right')

# 显示图形
plt.show()