import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 设置支持中文的字体和负号显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. 硬编码实验数据点 (基于 MicroBooNE NuMI 真实数据提取)
# 数据格式: [bin上界, 微分截面值 dsigma/dEe]
# 微分截面单位: 10^-39 cm^2 / GeV / nucleon
# 第一个 bin 的下界默认为 0.0
# =====================================================================
experimental_data = [
    [0.4, 2.72],   # Bin 1: [0.0, 0.4] GeV, dsigma/dEe = 2.72
    [0.8, 4.15],   # Bin 2: [0.4, 0.8] GeV, dsigma/dEe = 4.15
    [1.2, 2.98],   # Bin 3: [0.8, 1.2] GeV, dsigma/dEe = 2.98
    [1.6, 1.58],   # Bin 4: [1.2, 1.6] GeV, dsigma/dEe = 1.58
    [3.0, 0.28],   # Bin 5: [1.6, 3.0] GeV, dsigma/dEe = 0.28
    [6.0, 0.02]    # Bin 6: [3.0, 6.0] GeV, dsigma/dEe = 0.02
]

# 提取边界与微分截面值
edges = [0.0] + [row[0] for row in experimental_data]
diff_cross_sections = [row[1] for row in experimental_data]

print(">>> 原始实验数据 (微分截面):")
for i in range(len(diff_cross_sections)):
    print(f"  Bin {i+1}: [{edges[i]:.1f}, {edges[i+1]:.1f}] GeV, dsigma/dEe = {diff_cross_sections[i]:.4f}")

# =====================================================================
# 2. 计算原始数据的累积总截面 (Cumulative Total Cross Section)
# 累积截面 F(E) = \int_0^E (dsigma/dEe) dE
# 在每个原始边界处的值是离散积分值
# =====================================================================
cumulative_sigma = [0.0]
for i in range(len(diff_cross_sections)):
    width = edges[i+1] - edges[i]
    # 矩形积分积分值
    bin_integral = diff_cross_sections[i] * width
    cumulative_sigma.append(cumulative_sigma[-1] + bin_integral)

print("\n>>> 原始边界处的累积总截面 (10^-39 cm^2/nucleon):")
for i, E in enumerate(edges):
    print(f"  E = {E:.1f} GeV: Cumulative Sigma = {cumulative_sigma[i]:.4f}")

# =====================================================================
# 3. 使用三次多项式样条 (Cubic Spline) 插值
# 对离散的累积截面点 (edges, cumulative_sigma) 进行平滑插值
# 这样不仅保证在原始边界处物理总截面守恒，还能平滑地计算任意新边界的值
# =====================================================================
cs = CubicSpline(edges, cumulative_sigma, bc_type='clamped')

# =====================================================================
# 4. 重新分 bin: 从 0 开始，每 0.5 GeV 一个 bin，直到 6.0 GeV
# =====================================================================
new_bin_width = 0.5
new_edges = np.arange(0.0, 6.0 + new_bin_width, new_bin_width)

# 利用多项式插值计算新边界处的累积总截面值
new_cumulative_sigma = cs(new_edges)
# 物理约束：累积截面必须单调递增且非负
for i in range(1, len(new_cumulative_sigma)):
    if new_cumulative_sigma[i] < new_cumulative_sigma[i-1]:
        new_cumulative_sigma[i] = new_cumulative_sigma[i-1]

# 计算每个新 bin 内的总截面 (Total Cross Section)
# sigma_j = F(E_j) - F(E_{j-1})
new_total_cross_sections = []
for i in range(len(new_edges) - 1):
    val = new_cumulative_sigma[i+1] - new_cumulative_sigma[i]
    new_total_cross_sections.append(max(0.0, val))

print("\n>>> 重新分 bin (每 0.5 GeV) 后的每个 bin 内的总截面 (Total Cross Section):")
for i in range(len(new_total_cross_sections)):
    print(f"  Bin {i+1:02d}: [{new_edges[i]:.1f}, {new_edges[i+1]:.1f}] GeV, Total Sigma = {new_total_cross_sections[i]:.4f}")

# =====================================================================
# 5. 计算重新分 bin 后的微分截面，以便与输入进行对比
# (dsigma/dEe)_new = sigma_j / delta_E
# =====================================================================
new_diff_cross_sections = [sig / new_bin_width for sig in new_total_cross_sections]

# =====================================================================
# 6. 绘图与对比
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---- 子图 1: 总截面 (Total Cross Section) 阶梯图 ----
# 绘制阶梯图需要将数据对齐到 bin 的边界上
ax1.step(new_edges[:-1], new_total_cross_sections, where='post', color='teal', linewidth=2.5, label='重分 bin 后总截面 (0.5 GeV/bin)')
# 在每个 bin 的中心画点标出数值
bin_centers = new_edges[:-1] + new_bin_width / 2.0
ax1.scatter(bin_centers, new_total_cross_sections, color='darkorange', zorder=3, label='Bin 中心值')

ax1.set_title('重分 Bin 后的总截面阶梯图', fontsize=14, fontweight='bold')
ax1.set_xlabel('$E_e$ [GeV]', fontsize=12)
ax1.set_ylabel(r'$\sigma$ [$10^{-39}$ $\mathrm{cm}^2/\mathrm{nucleon}$]', fontsize=12)
ax1.set_xlim(0, 6.0)
ax1.set_ylim(bottom=0)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10)

# ---- 子图 2: 新旧微分截面 (Differential Cross Section) 对比 ----
# 绘制原始微分阶梯图
ax2.step(edges[:-1], diff_cross_sections, where='post', color='crimson', linestyle='-', linewidth=2.5, label='原始输入微分截面 (不等宽 bin)')
# 绘制重分 bin 后的微分阶梯图
ax2.step(new_edges[:-1], new_diff_cross_sections, where='post', color='royalblue', linestyle='--', linewidth=2, label='重分 bin 后推导微分截面 (0.5 GeV bin)')

# 绘制多项式插值得到的连续微分截面曲线（累积截面的导数）
fine_energies = np.linspace(0.0, 6.0, 500)
continuous_diff = cs.derivative()(fine_energies)
# 确保连续微分截面非负
continuous_diff = np.clip(continuous_diff, 0, None)
ax2.plot(fine_energies, continuous_diff, color='purple', alpha=0.5, linestyle=':', label='多项式插值连续曲线')

ax2.set_title('微分截面对比 (验证插值重构精度)', fontsize=14, fontweight='bold')
ax2.set_xlabel('$E_e$ [GeV]', fontsize=12)
ax2.set_ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}E_e}$ [$10^{-39}$ $\mathrm{cm}^2/\mathrm{GeV}/\mathrm{nucleon}$]', fontsize=12)
ax2.set_xlim(0, 6.0)
ax2.set_ylim(bottom=0)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.show()