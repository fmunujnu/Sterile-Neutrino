import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

# 1. 设定刚体的转动惯量 (I1 < I2 < I3，2轴为中间轴)
I1 = 0.005
I2 = 0.010
I3 = 0.015

# 2. 设定2轴的主转动角速度 Ω = 10π rad/s
omega_0 = 10 * np.pi

# 3. 设定1,3轴的轻微扰动范围
epsilon_orbit = np.logspace(-2, 0, 500)

periods = []
k_values = []

# 4. 循环计算每个扰动下的周期
for eps in epsilon_orbit:
    w1 = eps
    w2 = omega_0
    w3 = eps
    
    # 计算当前状态下的总能量 E 和角动量平方 L^2
    E = 0.5 * (I1 * w1**2 + I2 * w2**2 + I3 * w3**2)
    L2 = (I1 * w1)**2 + (I2 * w2)**2 + (I3 * w3)**2
    
    # k² 公式 (针对 I1 < I2 < I3 调整，保证 k² ∈ [0,1) 且 ε→0 时 k²→1)
    num_k2 = (I1 - I2) * (L2 - 2 * E * I3)
    den_k2 = (I2 - I3) * (2 * E * I1 - L2)
    k2 = num_k2 / den_k2

    k2 = np.clip(k2, 0, 1 - 1e-15)

    K_k = ellipk(k2)

    factor1 = 4 * K_k * np.sqrt(I2 * I3 / np.abs((I1 - I2) * (I1 - I3)))
    factor2 = np.sqrt(np.abs(I1 * (I1 - I2)) / (L2 - 2 * E * I2))
    T = factor1 * factor2
    
    periods.append(T)

# 转换为 numpy 数组方便处理
periods = np.array(periods)

# ====== 严格参考 inertia_scan.py 的全局绘图样式配置 ======
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Noto Serif SC', 'SimSun', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'         # 核心修复：完美解决对数坐标轴负号 '-' 显示错误问题
plt.rcParams['axes.unicode_minus'] = False

# 使用与参考代码高度一致的画布比例 (7, 4.5)
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

# 绘制对数曲线，线宽保持 2
ax.loglog(epsilon_orbit, periods, color='#333333', linewidth=2)
ax.set_xlim(1e-2, 1)
ax.set_ylim(np.min(periods) * 0.8, np.max(periods) * 1.5)

# 严格同步的字号、网格线 (:) 及透明度 (0.5) 设置
ax.set_xlabel(r'1,3轴扰动幅度 $\epsilon$ (rad$\cdot$s$^{-1}$)', fontsize=11)
ax.set_ylabel(r'旋转/翻转周期 $T$ (s)', fontsize=11)
ax.grid(True, which='both', linestyle=':', alpha=0.5)

# 统一的标题格式
plt.title('刚体翻转周期 vs 初始扰动 (中间轴)', fontsize=12, pad=12)
plt.tight_layout()

plt.savefig('period_vs_perturbation.png', bbox_inches='tight', dpi=300)
plt.show()