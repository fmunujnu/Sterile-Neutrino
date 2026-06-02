import numpy as np
import matplotlib.pyplot as plt

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# =====================================================================
# 【用户数据输入区】
# 请在此处填入您的 26 个原始数据值（自变量为能量，对应的因变量值可以是计数、强度等）
# 前 25 个值对应 0.1 GeV 到 2.5 GeV (步长 0.1 GeV)
# 第 26 个值对应 2.5 GeV 以上能量的平均数据
# =====================================================================
Y_RAW = [
        0.        , 0.03515726, 0.1230604, 0.17592975, 0.2216032, 0.27385078,
        0.30083114, 0.30912699, 0.32330211, 0.32699299, 0.3247482, 0.32804201,
        0.32729473, 0.33095618, 0.31724309, 0.31813633, 0.30701209, 0.31377639,
        0.29874747, 0.29447149, 0.29277409, 0.28382266, 0.28116426, 0.27507559,
        0.27508131, 0.25247578]

# =====================================================================
# 【参数配置区】
# =====================================================================
# 第 26 个 bin (> 2.5 GeV 平均值) 在拟合时所代表的等效能量中心点 (单位: GeV)
# 既然是 2.5 GeV 以上的平均值，其等效中心应大于 2.5 GeV，例如 3.0 或 3.5 GeV
ENERGY_26_CENTER = 5.0  

# 用于外推拟合的趋势参考点数。
# 我们利用前 25 个点中的最后 N 个点（例如最后 5 个点，对应 2.1-2.5 GeV）加上第 26 个点来拟合趋势。
# 这样可以确保外推线与前 25 个点的末尾完美、平滑地衔接。
N_TAIL_POINTS = 10

# 多项式拟合的阶数 (由于只拟合末尾的趋势进行外推，推荐 1 阶(线性) 或 2 阶)
POLY_DEGREE = 3


def main():
    # 确保输入数据长度正确
    if len(Y_RAW) != 26:
        raise ValueError(f"输入数据 Y_RAW 的长度必须为 26，当前长度为: {len(Y_RAW)}")
        
    y_orig = np.array(Y_RAW)

    # 1. 构建原始前 25 个 bin 对应的中心能量坐标
    x_normal = np.linspace(0.1, 2.5, 25)

    # 2. 提取用于趋势拟合的数据点（末尾的 N 个点 + 第 26 个平均值点）
    # 这样既能尊重高能平均值，又能完美顺接前 25 个点末端的趋势
    fit_x = np.append(x_normal[-N_TAIL_POINTS:], ENERGY_26_CENTER)
    fit_y = np.append(y_orig[25 - N_TAIL_POINTS:25], y_orig[25])

    # 3. 进行局部多项式拟合，用于预测 2.5 GeV 之后的数据趋势
    poly_coeffs = np.polyfit(fit_x, fit_y, deg=POLY_DEGREE)
    poly_func = np.poly1d(poly_coeffs)

    # 4. 拼接生成最终的 60 个 bin 的数据 (0.1 GeV 到 6.0 GeV)
    x_new = np.arange(0.1, 6.1, 0.1)  # 60 个能量点
    
    # 前 25 个点 (0.1 - 2.5 GeV) 完全保留原始真实值，不作任何修改
    y_new_part1 = y_orig[:25]
    
    # 后 35 个点 (2.6 - 6.0 GeV) 使用多项式拟合出的函数进行外推计算
    x_extrapolate = x_new[25:]  # 2.6, 2.7, ..., 6.0 GeV
    y_new_part2 = poly_func(x_extrapolate)
    
    # 物理合理性限制：如果因变量不应为负数，可在此处将外推部分的负值截断为 0
    y_new_part2 = np.clip(y_new_part2, 0, None)
    
    # 完美拼接
    y_new = np.concatenate([y_new_part1, y_new_part2])

    # 5. 在终端打印出可以直接复制使用的 NumPy 格式数组
    print("\n" + "="*50)
    print("【插值与外推后的数据输出 (0.1 - 6.0 GeV，共 60 个 bin)】")
    print("前 25 个数据保持 100% 原始真实值不变，仅对 2.5 GeV 之后的数据进行了外推插值。")
    print("您可以直接复制下方代码并在其他 Python 文件中使用：")
    print("="*50)
    
    # 格式化输出能量自变量 X 数组
    x_array_str = ", ".join([f"{val:.1f}" for val in x_new])
    print(f"energy_bins = np.array([\n    {x_array_str}\n])")
    print()
    
    # 格式化输出插值后的因变量 Y 数组（每 5 个数据换一行，方便阅读）
    y_formatted_lines = []
    for i in range(0, len(y_new), 5):
        chunk = y_new[i:i+5]
        chunk_str = ", ".join([f"{val:.6f}" for val in chunk])
        y_formatted_lines.append(f"    {chunk_str}")
    y_array_str = ",\n".join(y_formatted_lines)
    
    print(f"interpolated_values = np.array([\n{y_array_str}\n])")
    print("="*50 + "\n")

    # 6. 绘制对比阶梯图 (Step Plot)
    plt.figure(figsize=(12, 7), dpi=100)

    # 绘制插值与外推拼接后的完整数据 (0.1 - 6.0 GeV)
    # where='mid' 表示给定的能量点代表该 bin 的中心位置
    plt.step(x_new, y_new, where='mid', color='#1f77b4', linewidth=2, 
             label='插值与外推拼接后数据 (0.1 - 6.0 GeV)')
    plt.fill_between(x_new, y_new, step="mid", alpha=0.15, color='#1f77b4')

    # 绘制原始的前 25 个精细 bin 作为对比 (应当与新数据的前 25 个点完全重合)
    plt.step(x_normal, y_orig[:25], where='mid', color='#ff7f0e', linewidth=3, 
             linestyle='--', label='原始前 25 个精细 Bin (完全保留, 未作修改)')

    # 特殊标记第 26 个高能平均值数据点
    plt.scatter(ENERGY_26_CENTER, y_orig[25], color='red', s=120, zorder=5,
                label=f'第 26 个 Bin (平均值, 设等效中心为 {ENERGY_26_CENTER:.1f} GeV)')

    # 图表装饰与布局
    plt.title('能量区间局部拟合外推对比阶梯图 (前25个Bin保持原样)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('能量 (Energy / GeV)', fontsize=12)
    plt.ylabel('数值 / 测量值 (Value / Yield)', fontsize=12)
    plt.xlim(0.0, 6.2)
    
    # y轴范围自动适应数据，并预留10%的高度
    plt.ylim(min(y_orig.min(), y_new.min()) * 0.9, max(y_orig.max(), y_new.max()) * 1.1)
    
    # 绘制边界辅助线
    plt.axvline(x=2.5, color='purple', linestyle='-.', alpha=0.7, label='原始精细数据边界 (2.5 GeV)')
    plt.text(2.6, plt.ylim()[1] * 0.8, '保持原样区域 ➔', color='green', fontsize=11, fontweight='bold', ha='right')
    plt.text(2.6, plt.ylim()[1] * 0.75, '➔ 仅外推此区域', color='purple', fontsize=11, fontweight='bold', ha='left')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    main()