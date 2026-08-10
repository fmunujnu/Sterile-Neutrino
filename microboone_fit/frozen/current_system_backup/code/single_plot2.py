"""单次能谱图 —— 给定参数，仅绘制第一张图，不扫描。"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib.util as iu

# ── 用户参数（sin^2 2θ_μe = 0.003, Δm^2_41 = 1.2 eV^2, sin^2 θ_24 = 0.018） ──
_s2_24 = 0.018
_s2_mue = 0.003
_s2_14 = _s2_mue / (4.0 * _s2_24)
PARAMS = {
    'theta14': np.arcsin(np.sqrt(_s2_14)),
    'theta24': np.arcsin(np.sqrt(_s2_24)),
    'theta15': 0.0,
    'theta25': 0.0,
    'theta34': 0.0,
    'theta35': 0.0,
    'theta45': 0.0,
    'delta14': 0.0,
    'delta15': 0.0,
    'delta24': 0.0,
    'delta25': 0.0,
    'delta34': 0.0,
    'delta35': 0.0,
    'delta45': 0.0,
    'dm41': np.sqrt(1.2),
    'dm51': 0.0,
}

# ── 第二套参数（取消注释并填入你想要的参数即可启用） ─────────────────
PARAMS2 = {
    'theta14': 0.0,
    'theta24': 0.0,
    'theta15': 0.0,
    'theta25': 0.0,
    'theta34': 0.0,
    'theta35': 0.0,
    'theta45': 0.0,
    'delta14': 0.0,
    'delta15': 0.0,
    'delta24': 0.0,
    'delta25': 0.0,
    'delta34': 0.0,
    'delta35': 0.0,
    'delta45': 0.0,
    'dm41': 0.0,
    'dm51': 0.0,
}

# ── 加载主模块 ───────────────────────────────────────────────────────────
_base = os.path.dirname(__file__)
DATA_DIR = os.path.join(_base, '..', 'data')
OUTPUT_DIR = _base
sys.path.insert(0, _base)
spec = iu.spec_from_file_location(
    'main', os.path.join(_base, 'testchonggou2.py'))
m = iu.module_from_spec(spec)
spec.loader.exec_module(m)
rd = __import__('read_data')

exp = m.load_experiment()
outdir = OUTPUT_DIR
os.makedirs(outdir, exist_ok=True)

# ── 额外读取 CSV 的 Signal+Background ─────────────────────────────────────
csv_path = os.path.join(DATA_DIR, 'HEPData-ins3088922-v1-Unconstrained_14_channels.csv')
_, _, sigbkg_csv = rd.read_three_spectra(csv_path)
sigbkg_csv = sigbkg_csv[:104].reshape(4, 26)

# ── 绘图 ─────────────────────────────────────────────────────────────────
CH_NAMES = ['eCC FC', 'eCC PC', r'$\mu$CC FC', r'$\mu$CC PC']

bkg   = exp['background'].reshape(4, 26)
sig, _   = m.compute_spectrum(PARAMS, exp)
sig   = sig.reshape(4, 26)
total = bkg + sig
sig2, _ = m.compute_spectrum(PARAMS2, exp)
sig2   = sig2.reshape(4, 26)
total2 = bkg + sig2
data  = exp['observed'].reshape(4, 26)
E     = exp['obs_centers']
Elo   = exp['Elo']
Ehi   = exp['Ehi']
xerr  = [E - Elo, Ehi - E]

err_up = exp['observed_err_up'].reshape(4, 26)
err_dn = exp['observed_err_down'].reshape(4, 26)

E_step = np.append(Elo, Ehi[-1])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flat
for c in range(4):
    ax = axes[c]
    T_step = np.append(total[c], total[c][-1])
    B_step = np.append(bkg[c], bkg[c][-1])

    ax.step(E_step, B_step, where='post',
            color='steelblue', ls='--', lw=1.6, label='Background only')
    ax.step(E_step, T_step, where='post',
            color='steelblue', ls='-', lw=2., label='Bkg + prediction')
    T2_step = np.append(total2[c], total2[c][-1])
    ax.step(E_step, T2_step, where='post',
            color='darkorange', ls='-', lw=2., label='Null')

    ax.errorbar(E, data[c],
                yerr=[err_dn[c], err_up[c]],
                xerr=xerr,
                fmt='o',
                color='crimson',
                ms=5,
                capsize=3,
                elinewidth=1.2,
                markeredgewidth=0.8,
                label='Observed')

    S_step = np.append(sigbkg_csv[c], sigbkg_csv[c][-1])
    ax.step(E_step, S_step, where='post',
            color='seagreen', ls='-', lw=1.8, label='Sig+Bkg')

    ax.set_xlabel(r'Reconstructed $E_\nu^{\rm QE}\ \rm [GeV]$', fontsize=11)
    ax.set_ylabel('Event count', fontsize=11)
    ax.set_title(f'MicroBooNE — {CH_NAMES[c]}', fontsize=11)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=9)
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_xlim(Elo[0]*0.98, Ehi[-1]*1.02)
    ax.set_ylim(bottom=0.)

fig.suptitle('MicroBooNE — Prediction vs Data', fontsize=13, y=1.01)
fig.tight_layout()
if True:
    fig.savefig(os.path.join(outdir, 'microboone_spectrum_genie.png'), dpi=150, bbox_inches='tight')
    print(f'[single_plot2] saved to {os.path.join(outdir, "microboone_spectrum_genie.png")}')
