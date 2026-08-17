"""
读取 eArcrosssection.txt，对累积截面做多项式（三次样条）插值，
在 [0, 6] GeV 上均匀重分为 60 个 bin，输出单位 10^-39 cm^2/nucleon。

数据列（1-based 描述，文件中至少 5 列）:
  第 1 列: 序号（可选，不参与计算）
  第 2 列: 横向误差棒下界
  第 3 列: 横向误差棒上界
  第 4 列: 数据点横坐标 E [GeV]
  第 5 列: 纵坐标 σ [10^-38 cm^2/nucleon]
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_FILE = Path(__file__).resolve().parent / 'eArcrosssection.txt'
E_MIN, E_MAX = 0.0, 6.0
N_BINS = 60
BIN_WIDTH = (E_MAX - E_MIN) / N_BINS
UNIT_SCALE = 10.0  # 10^-38 -> 10^-39


def load_eAr_data(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f'数据文件为空或不存在: {path}\n'
            '请填入至少 5 列数据（第2–5列为 x_err_lo, x_err_hi, E, sigma）。'
        )

    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 5:
                continue
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue
            rows.append(nums)

    if not rows:
        raise ValueError(f'未在 {path} 中解析到数值数据行')

    data = np.array(rows)
    if data.shape[1] >= 5:
        x_err_lo = data[:, 1]
        x_err_hi = data[:, 2]
        x = data[:, 3]
        y_38 = data[:, 4]
    else:
        x_err_lo = data[:, 0]
        x_err_hi = data[:, 1]
        x = data[:, 2]
        y_38 = data[:, 3]

    order = np.argsort(x)
    x_err_lo = x_err_lo[order]
    x_err_hi = x_err_hi[order]
    x = x[order]
    y_38 = y_38[order]

    # 累积截面从原点起算
    if x[0] > E_MIN or y_38[0] != 0.0:
        x = np.insert(x, 0, E_MIN)
        y_38 = np.insert(y_38, 0, 0.0)
        x_err_lo = np.insert(x_err_lo, 0, E_MIN)
        x_err_hi = np.insert(x_err_hi, 0, E_MIN)

    y_39 = y_38 * UNIT_SCALE
    return x_err_lo, x_err_hi, x, y_39


def main():
    x_err_lo, x_err_hi, x, y = load_eAr_data(DATA_FILE)

    print('>>> 原始数据 (已换算为 10^-39 cm^2/nucleon):')
    for i in range(len(x)):
        print(f'  [{x_err_lo[i]:.4f}, {x_err_hi[i]:.4f}] GeV, E={x[i]:.4f}, sigma={y[i]:.6f}')

    # 三次样条多项式插值（分段三次 = 多项式插值）
    cs = CubicSpline(x, y, bc_type='clamped', extrapolate=False)

    uniform_edges = np.linspace(E_MIN, E_MAX, N_BINS + 1)
    x_rebin = uniform_edges[1:]

    def eval_sigma(xq):
        out = np.empty_like(xq, dtype=float)
        for i, xe in enumerate(xq):
            if xe <= x[-1]:
                out[i] = float(cs(xe))
            else:
                out[i] = y[-1]
        return out

    y_rebin = eval_sigma(x_rebin)
    y_rebin = np.maximum.accumulate(y_rebin)

    fine_x = np.linspace(E_MIN, E_MAX, 500)
    y_fine = eval_sigma(fine_x)
    y_fine = np.maximum.accumulate(y_fine)

    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print('\n>>> 最终 60-bin 重分结果 (10^-39 cm^2/nucleon, 单调递增):')
    print(np.array2string(y_rebin, separator=', '))
    print(f"  单调性检查: {'通过' if np.all(np.diff(y_rebin) >= 0) else '未通过'}")
    print(f'  sigma(E=6 GeV) = {y_rebin[-1]:.6f}')

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax0 = axes[0, 0]
    xerr = np.vstack([x - x_err_lo, x_err_hi - x])
    ax0.errorbar(x, y, xerr=xerr, fmt='o', color='crimson', ms=6, capsize=3,
                 elinewidth=1.2, label='原始数据')
    ax0.set_title('原始实验数据', fontsize=12, fontweight='bold')
    ax0.set_xlabel(r'$E$ [GeV]')
    ax0.set_ylabel(r'$\sigma$ [$10^{-39}$ cm$^2$/nucleon]')
    ax0.set_xlim(E_MIN, E_MAX)
    ax0.set_ylim(bottom=0)
    ax0.grid(True, ls='--', alpha=0.5)
    ax0.legend()

    ax1 = axes[0, 1]
    ax1.plot(fine_x, y_fine, color='royalblue', lw=2.5, label='三次样条插值')
    ax1.plot(x, y, 'o', color='crimson', ms=5, zorder=3, label='原始点')
    ax1.set_title('多项式（三次样条）插值曲线', fontsize=12, fontweight='bold')
    ax1.set_xlabel(r'$E$ [GeV]')
    ax1.set_ylabel(r'$\sigma$ [$10^{-39}$ cm$^2$/nucleon]')
    ax1.set_xlim(E_MIN, E_MAX)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, ls='--', alpha=0.5)
    ax1.legend()

    ax2 = axes[1, 0]
    ax2.plot(x, y, 'o-', color='crimson', ms=5, lw=1.5, alpha=0.8, label='原始')
    ax2.plot(x_rebin, y_rebin, 's--', color='teal', ms=4, lw=1.5, label='60-bin 重分')
    ax2.set_title('原始 vs 重分 bin 对比', fontsize=12, fontweight='bold')
    ax2.set_xlabel(r'$E$ [GeV]')
    ax2.set_ylabel(r'$\sigma$ [$10^{-39}$ cm$^2$/nucleon]')
    ax2.set_xlim(E_MIN, E_MAX)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, ls='--', alpha=0.5)
    ax2.legend()

    ax3 = axes[1, 1]
    ax3.plot(x_rebin, y_rebin, 'o-', color='teal', lw=2, ms=4,
             label=r'60-bin $\sigma(E)$')
    ax3.step(x_rebin, y_rebin, where='pre', color='darkorange', lw=1.5,
             ls='--', alpha=0.8, label='阶梯展示')
    ax3.set_title('最终 60-bin 数组', fontsize=12, fontweight='bold')
    ax3.set_xlabel(r'$E$ [GeV]')
    ax3.set_ylabel(r'$\sigma$ [$10^{-39}$ cm$^2$/nucleon]')
    ax3.set_xlim(E_MIN, E_MAX)
    ax3.set_ylim(bottom=0)
    ax3.grid(True, ls='--', alpha=0.5)
    ax3.legend()

    fig.suptitle(
        r'eAr 截面: 三次样条插值 $\rightarrow$ [0,6] GeV 均匀 60 bin ($10^{-39}$)',
        fontsize=14, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
