import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Noto Serif SC', 'SimSun', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

I1 = 5.0
I3 = 1.0
omega_0 = 1.0
eps = 1e-4

theta = np.linspace(0.005, np.pi / 2 - 0.005, 5000)
alpha_series = np.sin(theta) ** 2
I2_series = I3 + (I1 - I3) * alpha_series

periods = []

for I2 in I2_series:
    w1 = eps
    w2 = omega_0
    w3 = eps

    E = 0.5 * (I1 * w1**2 + I2 * w2**2 + I3 * w3**2)
    L2 = (I1 * w1)**2 + (I2 * w2)**2 + (I3 * w3)**2

    # ====== 核心修复：解析延拓与严格分支求解 ======
    # 显式计算跨越分轨的流形标志位 Delta
    Delta = 2 * E * I2 - L2
    
    # 严格根据哈密顿系统的几何流形进行无缝分轨计算
    if Delta >= 0:
        # 状态 A：未越过分轨
        k2 = (I1 - I2) * (2 * E * I3 - L2) / ((I1 - I3) * (2 * E * I2 - L2))
        k2 = np.clip(k2, 0, 1 - 1e-15)
        T = 4 * ellipk(k2) * np.sqrt((I1 * I3) / ((I1 - I2) * (L2 - 2 * E * I3)))
    else:
        # 状态 B：越过分轨，进行解析延拓
        k2 = (I2 - I3) * (2 * E * I1 - L2) / ((I1 - I3) * (2 * E * I2 - L2))
        k2 = np.clip(k2, 0, 1 - 1e-15)
        T = 4 * ellipk(k2) * np.sqrt((I1 * I3) / ((I2 - I3) * (2 * E * I1 - L2)))
    # =============================================

    periods.append(T)

periods = np.array(periods)

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

# 严格保留你原本的所有绘图设置、颜色与格式
ax.plot(alpha_series, periods, color='#333333', linewidth=2)
ax.set_yscale('log')
ax.set_xlabel(r'无量纲中间惯量系数 $\alpha = (I_2 - I_3)/(I_1 - I_3)$', fontsize=11)
ax.set_ylabel(r'翻转周期 $T$ (s)', fontsize=11)
ax.grid(True, which='both', linestyle=':', alpha=0.5)

ax.axvline(x=0, color='gray', linestyle='-.', alpha=0.5)
ax.axvline(x=1, color='gray', linestyle='-.', alpha=0.5)

plt.title(r'中间惯量 $I_2$ 对翻转周期 $T$ 的影响 ($\epsilon = 10^{-4}$)', fontsize=12, pad=12)
plt.tight_layout()

plt.savefig('inertia_sensitivity_analysis.png', bbox_inches='tight')
plt.show()