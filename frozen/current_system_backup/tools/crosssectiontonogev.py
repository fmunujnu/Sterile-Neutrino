import numpy as np

# 1. 原始截面数据
crosssection_e = np.array([
    0.21670000, 0.71202077, 0.95278943, 1.03394671, 1.05043330, 1.08050530,
    1.12029361, 1.14546764, 1.15773137, 1.16635675, 1.17281313, 1.17587717,
    1.17595653, 1.17406194, 1.17063597, 1.16586087, 1.15991962, 1.15299872,
    1.14530825, 1.13706253, 1.12838231, 1.11933295, 1.10999680, 1.10046657,
    1.09081527, 1.08110041, 1.07136385, 1.06163546, 1.05194947, 1.04234404,
    1.03285359, 1.02350798, 1.01431393, 1.00525507, 0.99634134, 0.98761347,
    0.97909080, 0.97076390, 0.96262707, 0.95468224, 0.94693150, 0.93937624,
    0.93201445, 0.92483867, 0.91785393, 0.91108864, 0.90455017, 0.89819926,
    0.89200568, 0.88597119, 0.88010577, 0.87442897, 0.86895207, 0.86366067,
    0.85853633, 0.85356133, 0.84872529, 0.84404120, 0.83952685, 0.83520000
])

crosssection_ebar = np.array([
    0.00950000, 0.04791136, 0.08018255, 0.10748477, 0.13098922, 0.15173652,
    0.17012444, 0.18635767, 0.20066458, 0.21327856, 0.22441426, 0.23427335,
    0.24301698, 0.25078336, 0.25766400, 0.26373458, 0.26911695, 0.27394642,
    0.27826506, 0.28206985, 0.28542673, 0.28843891, 0.29114981, 0.29356359,
    0.29569511, 0.29756833, 0.29921961, 0.30069391, 0.30201522, 0.30318904,
    0.30422053, 0.30511587, 0.30589957, 0.30661399, 0.30726414, 0.30781232,
    0.30825658, 0.30864219, 0.30899317, 0.30930022, 0.30955543, 0.30975660,
    0.30990943, 0.31003173, 0.31013513, 0.31021793, 0.31027592, 0.31030268,
    0.31030208, 0.31030105, 0.31031042, 0.31029485, 0.31023896, 0.31019848,
    0.31020679, 0.31019699, 0.31011331, 0.30999605, 0.30990501, 0.30990000
])

crosssection_mu = np.array([
    0.00000000, 0.09081792, 0.26961960, 0.48229866, 0.67474870, 0.80454862,
    0.88679484, 0.95015062, 1.00087166, 1.03815915, 1.06464000, 1.08394270,
    1.09774930, 1.10693290, 1.11235744, 1.11485463, 1.11500853, 1.11326846,
    1.10998578, 1.10545318, 1.09990902, 1.09355709, 1.08657747, 1.07913078,
    1.07132194, 1.06321601, 1.05488585, 1.04641031, 1.03784986, 1.02924850,
    1.02064288, 1.01206361, 1.00355049, 0.99515227, 0.98688813, 0.97874329,
    0.97071599, 0.96282312, 0.95508189, 0.94750821, 0.94010774, 0.93287116,
    0.92580134, 0.91892315, 0.91224571, 0.90574601, 0.89940819, 0.89323733,
    0.88723771, 0.88140647, 0.87574284, 0.87025348, 0.86493760, 0.85977178,
    0.85474154, 0.84986971, 0.84517580, 0.84064633, 0.83626112, 0.83200000
])

crosssection_mubar = np.array([
    0.00000000, 0.00870031, 0.03393395, 0.06751907, 0.10127383, 0.12855927,
    0.15033089, 0.16944933, 0.18643880, 0.20111159, 0.21378128, 0.22491330,
    0.23472454, 0.24333455, 0.25090702, 0.25761811, 0.26357788, 0.26886331,
    0.27355313, 0.27772459, 0.28142656, 0.28469298, 0.28758475, 0.29017803,
    0.29250974, 0.29458879, 0.29643736, 0.29808778, 0.29955600, 0.30084411,
    0.30197013, 0.30296718, 0.30385502, 0.30463894, 0.30533024, 0.30594754,
    0.30649712, 0.30696963, 0.30736834, 0.30771497, 0.30801951, 0.30827235,
    0.30847420, 0.30864537, 0.30879851, 0.30892915, 0.30903389, 0.30911497,
    0.30917292, 0.30920297, 0.30920616, 0.30919911, 0.30919637, 0.30920082,
    0.30920810, 0.30919831, 0.30915716, 0.30909786, 0.30903920, 0.30900000
])

# 2. 构造能量区间中点数组 (使用简化后的实现)
energy_midpoints = np.arange(0.00, 3.00, 0.05) + 0.025

# 3. 每一个数据乘上它所对应的中点能量
converted_e = crosssection_e * energy_midpoints
converted_ebar = crosssection_ebar * energy_midpoints
converted_mu = crosssection_mu * energy_midpoints
converted_mubar = crosssection_mubar * energy_midpoints

# 4. 打印转换结果函数 (严格匹配原格式与变量名输出)
def print_array(name, arr):
    print(f"\n# --- {name} (乘能量后的结果) ---")
    print(f"{name} = np.array([")
    formatted_rows = []
    for i in range(0, len(arr), 6):
        chunk = arr[i:i+6]
        line = "    " + ", ".join([f"{val:.8f}" for val in chunk])
        formatted_rows.append(line)
    print(",\n".join(formatted_rows))
    print("])")

# 按原数组名输出结果到控制台
print_array("crosssection_e", converted_e)
print_array("crosssection_ebar", converted_ebar)
print_array("crosssection_mu", converted_mu)
print_array("crosssection_mubar", converted_mubar)

# 5. 实现绘图进行对比
try:
    import matplotlib.pyplot as plt
    
    # 创建包含两个子图的画布（左边原始截面，右边乘以能量后的截面）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制原始截面
    ax1.plot(energy_midpoints, crosssection_e, label=r'$\nu_e$ CC', color='blue', linewidth=2)
    ax1.plot(energy_midpoints, crosssection_ebar, label=r'$\bar{\nu}_e$ CC', color='cyan', linewidth=2)
    ax1.plot(energy_midpoints, crosssection_mu, label=r'$\nu_\mu$ CC', color='red', linewidth=2)
    ax1.plot(energy_midpoints, crosssection_mubar, label=r'$\bar{\nu}_\mu$ CC', color='orange', linewidth=2)
    ax1.set_title("Original Cross Sections", fontsize=14)
    ax1.set_xlabel("Energy E (GeV)", fontsize=12)
    ax1.set_ylabel(r"$\sigma$ ($10^{-38}$ $cm^2$)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)
    
    # 绘制乘以能量后的截面
    ax2.plot(energy_midpoints, converted_e, label=r'$\nu_e$ CC $\times$ E', color='blue', linestyle='--', linewidth=2)
    ax2.plot(energy_midpoints, converted_ebar, label=r'$\bar{\nu}_e$ CC $\times$ E', color='cyan', linestyle='--', linewidth=2)
    ax2.plot(energy_midpoints, converted_mu, label=r'$\nu_\mu$ CC $\times$ E', color='red', linestyle='--', linewidth=2)
    ax2.plot(energy_midpoints, converted_mubar, label=r'$\bar{\nu}_\mu$ CC $\times$ E', color='orange', linestyle='--', linewidth=2)
    ax2.set_title("Scaled Cross Sections ($\sigma \times E$)", fontsize=14)
    ax2.set_xlabel("Energy E (GeV)", fontsize=12)
    ax2.set_ylabel(r"$\sigma \times E$ ($10^{-38}$ $cm^2 \cdot$ GeV)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("\n提示: 未安装 matplotlib，无法进行绘图。但上方转换后的 np.array 数据已正确输出。")