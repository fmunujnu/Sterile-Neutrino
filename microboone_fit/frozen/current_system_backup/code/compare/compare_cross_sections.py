"""对比 mycrossection.txt (旧) 与 xsec_ccqe_Ar40_data.txt (GENIE CCQE) 的截面。"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

BASE = os.path.dirname(__file__)

# ── 1. 读取旧截面 (60 bins, 0.05 GeV spacing, 10^-38 cm²/nucleon) ─────
old_ns = {}
with open(os.path.join(BASE, 'mycrossection.txt'), 'r') as f:
    lines = f.readlines()

for name in ['crosssection_e', 'crosssection_ebar', 'crosssection_mu', 'crosssection_mubar']:
    vals = []
    collecting = False
    for line in lines:
        if name in line:
            collecting = True
            continue
        if collecting:
            cleaned = line.strip().rstrip(',').strip(']')
            for v in cleaned.split(','):
                v = v.strip()
                if v and not v.startswith(')') and not v.startswith(']'):
                    try:
                        vals.append(float(v.replace(']))', '').replace('])', '')))
                    except ValueError:
                        pass
            if '])' in line or ']))' in line:
                break
    old_ns[name] = np.array(vals)

E_old = np.linspace(0.025, 2.975, 60)   # 0.05 GeV spacing, centres

# ── 2. 读取 GENIE CCQE (60点 log-spaced, 10^-38 cm²/Ar-40 nucleus) ──
genie = np.loadtxt(os.path.join(BASE, 'xsec_ccqe_Ar40_data.txt'), comments='#',
                   dtype={'names': ('E', 'nu_mu', 'anti_nu_mu', 'nu_e', 'anti_nu_e'),
                          'formats': ('f8', 'f8', 'f8', 'f8', 'f8')})
E_genie = genie['E']

# 插值到旧截面能量网格, 并转换为单位: per nucleus → per nucleon (÷40)
interp = lambda col: interp1d(E_genie, genie[col], kind='cubic', fill_value='extrapolate')(E_old) / 40.0

genie_ns = {
    'crosssection_e':     interp('nu_e'),
    'crosssection_ebar':  interp('anti_nu_e'),
    'crosssection_mu':    interp('nu_mu'),
    'crosssection_mubar': interp('anti_nu_mu'),
}

# ── 3. 绘制对比 ─────────────────────────────────────────────────────────
labels = {
    'crosssection_e':     r'$\nu_e$',
    'crosssection_ebar':  r'$\bar{\nu}_e$',
    'crosssection_mu':    r'$\nu_\mu$',
    'crosssection_mubar': r'$\bar{\nu}_\mu$',
}
colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flat
for c, (name, label) in enumerate(labels.items()):
    ax = axes[c]
    ax.plot(E_old, old_ns[name], 'o-', ms=3, lw=1.2, color='steelblue', label='Old')
    ax.plot(E_old, genie_ns[name], 's-', ms=3, lw=1.2, color='crimson', label='GENIE CCQE (÷40)')
    ax.set_xlabel('E [GeV]', fontsize=11)
    ax.set_ylabel(r'$\sigma$ [$10^{-38}$ cm$^2$/nucleon]', fontsize=11)
    ax.set_title(f'CCQE Cross Section — {label}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3.0)

fig.suptitle('Cross-Section Comparison: Old vs GENIE v3.06 CCQE on Ar-40', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(BASE, 'cross_section_comparison.png'), dpi=150, bbox_inches='tight')
print(f'Saved: {os.path.join(BASE, "cross_section_comparison.png")}')
