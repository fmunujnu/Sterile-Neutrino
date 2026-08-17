import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. 实验数据：阶梯函数
#    第 1 列 = bin 上界 x；第 2 列 = 该 bin 内常数值
#    第 1 个 bin 下界 = 0；相邻 bin 共享边界（上界 = 下一 bin 下界）
# =====================================================================
experimental_data = np.array([
    [0.300, 2.891],
    [0.475, 5.385],
    [0.700, 3.982],
    [1.000, 2.666],
    [1.427, 1.350],
    [3.000, 0.259],
    [6.000, 0.069],
])

E_MIN, E_MAX = 0.0, 6.0
N_BINS = 60
BIN_WIDTH = (E_MAX - E_MIN) / N_BINS

# 原始阶梯边界与常数值
edges_orig = np.concatenate(([E_MIN], experimental_data[:, 0]))
values_orig = experimental_data[:, 1]
centers_orig = 0.5 * (edges_orig[:-1] + edges_orig[1:])

print(">>> 原始阶梯数据 (dsigma/dEe, 10^-39 cm^2/GeV/nucleon):")
for i in range(len(values_orig)):
    print(f"  Bin {i + 1}: [{edges_orig[i]:.3f}, {edges_orig[i + 1]:.3f}] GeV  ->  {values_orig[i]:.4f}")

# =====================================================================
# 2. [0, 6] 上均匀 60 个 bin，对阶梯函数做分段线性插值
#    在原始 bin 中心处取常数值，再线性插值到新 bin 边界
# =====================================================================
uniform_edges = np.linspace(E_MIN, E_MAX, N_BINS + 1)
f_at_edges = np.interp(
    uniform_edges,
    centers_orig,
    values_orig,
    left=values_orig[0],
    right=values_orig[-1],
)

# 每个新 bin 内的线性插值函数：端点值 f_at_edges[i], f_at_edges[i+1]
# bin 中心处的插值代表值（用于展示）
centers_uniform = 0.5 * (uniform_edges[:-1] + uniform_edges[1:])
f_at_centers = 0.5 * (f_at_edges[:-1] + f_at_edges[1:])

# =====================================================================
# 3. 对插值后的分段线性函数在 [0, 6] 上积分 → 累积截面 F(E)
#    F(E) = ∫_0^E f_lin(E') dE'
# =====================================================================
cumulative_F = np.zeros(N_BINS + 1)
for i in range(N_BINS):
    f_left, f_right = f_at_edges[i], f_at_edges[i + 1]
    cumulative_F[i + 1] = cumulative_F[i] + 0.5 * (f_left + f_right) * BIN_WIDTH

# =====================================================================
# 4. 对积分结果 F(E) 在 60 个均匀 bin 上分 bin
#    取每个 bin 右边界处的累积截面 F(E_{i+1})，长度 60，物理上单调递增
# =====================================================================
final_array = cumulative_F[1:].copy()
final_array = np.maximum.accumulate(final_array)

np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("\n>>> 最终 60-bin 积分结果 (累积截面 F(E), 单调递增):")
print(np.array2string(final_array, separator=', '))
print(f"  单调性检查: {'通过' if np.all(np.diff(final_array) >= 0) else '未通过'}")
print(f"  总截面 F(6 GeV) = {final_array[-1]:.6f}  (10^-39 cm^2/nucleon)")

# =====================================================================
# 5. 绘图：原始 / 插值 / 积分 / 最终结果
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ---- (1) 原始阶梯数据 ----
ax0 = axes[0, 0]
ax0.step(edges_orig[:-1], values_orig, where='post', color='crimson', lw=2,
         label='原始阶梯 (实验 bin)')
ax0.set_title('原始阶梯数据', fontsize=12, fontweight='bold')
ax0.set_xlabel(r'$E$ [GeV]')
ax0.set_ylabel(r'$\mathrm{d}\sigma/\mathrm{d}E$ [$10^{-39}$ cm$^2$/GeV/nucleon]')
ax0.set_xlim(E_MIN, E_MAX)
ax0.set_ylim(bottom=0)
ax0.grid(True, ls='--', alpha=0.5)
ax0.legend()

# ---- (2) 线性插值后的 60-bin 分段线性函数 ----
ax1 = axes[0, 1]
ax1.step(uniform_edges[:-1], f_at_edges[:-1], where='post', color='royalblue', lw=2,
         label='插值后 (60 bin, 边界值)')
ax1.plot(uniform_edges, f_at_edges, 'o', color='royalblue', ms=3, alpha=0.6)
ax1.set_title('线性插值结果 (60 均匀 bin)', fontsize=12, fontweight='bold')
ax1.set_xlabel(r'$E$ [GeV]')
ax1.set_ylabel(r'$\mathrm{d}\sigma/\mathrm{d}E$ [$10^{-39}$ cm$^2$/GeV/nucleon]')
ax1.set_xlim(E_MIN, E_MAX)
ax1.set_ylim(bottom=0)
ax1.grid(True, ls='--', alpha=0.5)
ax1.legend()

# ---- (3) 积分结果：累积截面 F(E) ----
ax2 = axes[1, 0]
ax2.plot(uniform_edges, cumulative_F, color='darkgreen', lw=2.5, marker='.', ms=4,
         label=r'累积积分 $F(E)=\int_0^E f\,\mathrm{d}E$')
ax2.set_title('插值函数积分 (累积截面)', fontsize=12, fontweight='bold')
ax2.set_xlabel(r'$E$ [GeV]')
ax2.set_ylabel(r'$F(E)$ [$10^{-39}$ cm$^2$/nucleon]')
ax2.set_xlim(E_MIN, E_MAX)
ax2.set_ylim(bottom=0)
ax2.grid(True, ls='--', alpha=0.5)
ax2.legend()

# ---- (4) 最终 60-bin：积分后的累积截面（单调递增）----
ax3 = axes[1, 1]
ax3.plot(uniform_edges[1:], final_array, 'o-', color='teal', lw=2, ms=4,
         label=r'$F(E)$ @ 各 bin 右边界')
ax3.step(uniform_edges[1:], final_array, where='pre', color='darkorange', lw=1.5,
         ls='--', alpha=0.8, label='阶梯展示')
ax3.set_title('最终 60-bin 积分结果 (单调递增)', fontsize=12, fontweight='bold')
ax3.set_xlabel(r'$E$ [GeV]')
ax3.set_ylabel(r'$F(E)$ [$10^{-39}$ cm$^2$/nucleon]')
ax3.set_xlim(E_MIN, E_MAX)
ax3.set_ylim(bottom=0)
ax3.grid(True, ls='--', alpha=0.5)
ax3.legend()

fig.suptitle('阶梯函数 → 线性插值 → 积分 → 60-bin 累积截面', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
