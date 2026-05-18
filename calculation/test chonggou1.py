import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans'],
    'mathtext.fontset': 'dejavusans', 'axes.unicode_minus': False,
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'figure.dpi': 100,
})

# ── Standard 3-flavor PMNS constants ────────────────────────────────────────
TH12, TH13, TH23, D13 = 33*np.pi/180, 8.6*np.pi/180, 49*np.pi/180, 0.0
M2_STD = np.array([0., 7.5e-5, 2.5e-3])   # eV²

# ── Analysis switches ────────────────────────────────────────────────────────
MODEL      = '1+3+1'   # '3'  |  '3+1'  |  '3+2'  |  '1+3+1'
SCAN_MODE  = 'fixed'   # 'fixed'  |  'profile'
RUN_GLOBAL = True

# 这个true 是干啥的


# ── Parameter names for 1+3+1 / 3+2 (5-neutrino) models ────────────────────
PARAM_NAMES = [
    'theta14','theta24','theta34','theta15','theta25','theta35','theta45',
    'delta14','delta24','delta34','delta15','delta25','delta35','delta45',
    'dm41','dm51',
]

# ── Parameter config: initial value, optimizer bounds, log/linear scale ───────
_z = dict(init=0., lo=0., hi=0., sc='linear')   # 复制了一份 _z 字典，从而快速将它们全部设为固定零值
PARAM_CFG = {
    'dm41'   : dict(init=0.87, lo=0.01, hi=100.,sc='log'),
    'dm51'   : dict(init=0.47, lo=0.01, hi=100.,sc='log'),
    'theta14': dict(init=0.15, lo=1e-4, hi=1.0, sc='log'),
    'theta15': dict(init=0.13, lo=1e-4, hi=1.0, sc='log'),
    'theta24': dict(init=0.13, lo=1e-4, hi=1.0, sc='log'),
    'theta25': dict(init=0.17, lo=1e-4, hi=1.0, sc='log'),
    **{k: dict(**_z) for k in ['theta34','theta35','theta45',
                                'delta14','delta15','delta24','delta25',
                                'delta34','delta35','delta45']},
}
#确定区间 初值 fix

# ── Coupling functions: map virtual scan axes to real parameters ─────────────
def _couple_mode1(p, v1, v2):
    """sin(th14)*sin(th24) and sin(th15)*sin(th25) axes."""
    p['theta14'] = p['theta24'] = float(np.clip(np.sqrt(max(v1, 0.)), 0., 1.))
    p['theta15'] = p['theta25'] = float(np.clip(np.sqrt(max(v2, 0.)), 0., 1.))

def _couple_mode3_perfect_strict(p, v1, v2):
    ##########本段令14/24相等 没有考虑5
    """
    禁止小角近似，严格求解: v1 = 4 * sin⁴θ * (1 - sin²θ)
    满足条件: theta14 = theta24 = theta
    """
    # 1. 定义三次方程的系数: 4x³ - 4x² + 0x + v1 = 0
    # 注意：我们找的是 x = sin²θ
    coeffs = [4.0, -4.0, 0.0, float(v1)]
    
    # 2. 求根
    roots = np.roots(coeffs)
    
    # 3. 筛选物理合理的根
    # 物理上 sin²θ 必须是实数，且在 [0, 1] 之间。
    # 对于该特定方程，在 v1 较小时，通常有一个极小的正实根。
    real_roots = roots[np.isreal(roots)].real
    valid_roots = real_roots[(real_roots >= 0) & (real_roots <= 1)]
    
    if len(valid_roots) > 0:
        # 取最小的正根（对应物理上的小混合角分支）
        st2 = np.min(valid_roots)
    else:
        # 如果 v1 超出物理极限 (16/27)，则做截断处理

        print(f"error: couple_mode3_perfect_strict")

    # 4. 赋值
    st = float(np.arcsin(np.sqrt(st2)))
    p['theta14'] = p['theta24'] = st
    p['dm41'] = float(v2)

_zero_roles = {k: 0. for k in ['theta34','theta35','theta45',
                                 'delta14','delta15','delta24','delta25',
                                 'delta34','delta35','delta45']}

# ── Scan task definitions ─────────────────────────────────────────────────────
SCAN_TASKS = [
    dict(
        ax1='_prod14',  ax2='_prod15',
        grid1=np.logspace(np.log10(1e-4),        np.log10(np.sqrt(1.0)), 50),
        grid2=np.logspace(np.log10(1e-4),        np.log10(np.sqrt(1.0)), 50),
        xlabel=r'$\sin\theta_{14}\cdot\sin\theta_{24}$',
        ylabel=r'$\sin\theta_{15}\cdot\sin\theta_{25}$',
        title =r'$\sin\theta_{14}\sin\theta_{24}$ vs $\sin\theta_{15}\sin\theta_{25}$',
        xscale='log', yscale='log', savepath='miniboone_mode1.png',
        extra_fixed=dict(dm41=0.87, dm51=0.47),
        couple=_couple_mode1,
        param_roles={**_zero_roles, 'dm41':'fixed_init', 'dm51':'fixed_init'},
        proj_fn=lambda p: (p.get('theta14',0.)*p.get('theta24',0.),
                           p.get('theta15',0.)*p.get('theta25',0.)),
    ),
    dict(
        ax1='dm41',  ax2='dm51',
        grid1=np.logspace(np.log10(0.01), np.log10(100.), 50),
        grid2=np.logspace(np.log10(0.01), np.log10(100.), 50),
        xlabel=r'$|\Delta m^2_{41}|\ \rm[eV^2]$',
        ylabel=r'$|\Delta m^2_{51}|\ \rm[eV^2]$',
        title =r'$|\Delta m^2_{41}|$ vs $|\Delta m^2_{51}|$',
        xscale='log', yscale='log', savepath='miniboone_mode2.png',
        extra_fixed=dict(theta14=0.15, theta24=0.13, theta15=0.13, theta25=0.17),
        couple=None,
        param_roles={**_zero_roles,
                     'theta14':'free','theta24':'free','theta15':'free','theta25':'free'},
        proj_fn=lambda p: (abs(p.get('dm41',0.)), abs(p.get('dm51',0.))),
    ),
    dict(
        ax1='_sin2_2theta_eff',  ax2='dm41',
        grid1=np.logspace(np.log10(1e-4), np.log10(1.0),  100),
        grid2=np.logspace(np.log10(0.01),  np.log10(100.), 100),
        xlabel=r'$\sin^2(2\theta_{\mu e}^{\rm eff})$',
        ylabel=r'$|\Delta m^2_{41}|\ \rm[eV^2]$',
        title =r'$\sin^2(2\theta_{\mu e}^{\rm eff})$ vs $|\Delta m^2_{41}|$',
        xscale='log', yscale='log', savepath='miniboone_mode3.png',
        extra_fixed=dict(dm51=0.47, theta15=0.13, theta25=0.17),
        couple=_couple_mode3,
        param_roles={**_zero_roles,
                     'dm51':'fixed_init','theta15':'fixed_init','theta25':'fixed_init'},
        proj_fn=lambda p: (4.*p.get('theta14',0.)**2*p.get('theta24',0.)**2,
                           abs(p.get('dm41',0.))),
    ),
]###改变扫描精度和范围 直接调整坐标轴

# ═══════════════════════════════════════════════════════════════════════════
# Part 1 · Experimental data
# ═══════════════════════════════════════════════════════════════════════════

def load_experiment():
    """
    Hard-code and return all MiniBooNE detector/beam quantities in a dict.

    Keys
    ----
    E_true, L_km           – true energy bins and baseline
    Elo, Ehi, obs_centers  – observed energy bin edges and centres (GeV)
    observed, background   – event counts (8 bins)
    covariance             – 8×8 fractional covariance matrix
    response_matrix        – (n_true × 8) smearing matrix
    precomp_nu, precomp_nub– pre-multiplied flux weights (numu, numub)
    """
    # ── Flux bins: 200 bins × 0.05 GeV ──────────────────────────────────────
    Elo = np.arange(0.00, 10.00, 0.05)
    Ehi = Elo + 0.05
    E_centers = (Elo + Ehi) / 2.

    numu_flux = np.array([
        2.272e-12,8.566e-12,1.112e-11,1.335e-11,1.658e-11,1.820e-11,1.946e-11,2.045e-11,2.161e-11,2.241e-11,
        2.279e-11,2.292e-11,2.275e-11,2.253e-11,2.214e-11,2.156e-11,2.078e-11,1.992e-11,1.894e-11,1.789e-11,
        1.677e-11,1.558e-11,1.439e-11,1.318e-11,1.193e-11,1.069e-11,9.503e-12,8.356e-12,7.278e-12,6.292e-12,
        5.396e-12,4.601e-12,3.902e-12,3.285e-12,2.760e-12,2.312e-12,1.932e-12,1.616e-12,1.355e-12,1.138e-12,
        9.589e-13,8.150e-13,6.928e-13,5.937e-13,5.147e-13,4.478e-13,3.935e-13,3.500e-13,3.150e-13,2.867e-13,
        2.615e-13,2.409e-13,2.273e-13,2.110e-13,1.995e-13,1.920e-13,1.815e-13,1.726e-13,1.665e-13,1.601e-13,
        1.554e-13,1.493e-13,1.442e-13,1.412e-13,1.363e-13,1.323e-13,1.265e-13,1.217e-13,1.183e-13,1.140e-13,
        1.102e-13,1.060e-13,1.014e-13,9.700e-14,9.340e-14,9.001e-14,8.641e-14,8.190e-14,7.867e-14,7.464e-14,
        7.146e-14,6.812e-14,6.499e-14,6.185e-14,5.858e-14,5.614e-14,5.320e-14,5.016e-14,4.765e-14,4.561e-14,
        4.281e-14,4.087e-14,3.841e-14,3.632e-14,3.432e-14,3.263e-14,3.016e-14,2.857e-14,2.689e-14,2.529e-14,
        2.372e-14,2.227e-14,2.103e-14,1.957e-14,1.834e-14,1.730e-14,1.615e-14,1.513e-14,1.406e-14,1.303e-14,
        1.214e-14,1.129e-14,1.047e-14,9.569e-15,8.870e-15,8.148e-15,7.429e-15,6.765e-15,6.097e-15,5.492e-15,
        4.977e-15,4.445e-15,3.967e-15,3.492e-15,3.037e-15,2.595e-15,2.225e-15,1.854e-15,1.537e-15,1.220e-15,
        9.780e-16,7.842e-16,6.198e-16,4.786e-16,3.334e-16,1.971e-16,9.391e-17,2.738e-17,6.065e-18,4.135e-18,
        1.933e-18,9.888e-19,4.494e-20,*([0.0]*57)])

    numub_flux = np.array([
        2.560e-12,5.671e-12,3.300e-12,2.028e-12,1.623e-12,1.395e-12,1.301e-12,1.249e-12,1.171e-12,1.054e-12,
        9.580e-13,8.695e-13,8.098e-13,7.434e-13,6.910e-13,6.314e-13,5.905e-13,5.504e-13,5.079e-13,4.708e-13,
        4.347e-13,4.021e-13,3.703e-13,3.443e-13,3.173e-13,2.872e-13,2.597e-13,2.337e-13,2.101e-13,1.903e-13,
        1.718e-13,1.507e-13,1.341e-13,1.173e-13,1.053e-13,9.241e-14,8.188e-14,7.115e-14,6.349e-14,5.547e-14,
        4.799e-14,4.071e-14,3.592e-14,3.082e-14,2.638e-14,2.248e-14,1.878e-14,1.623e-14,1.391e-14,1.162e-14,
        1.010e-14,8.691e-15,7.382e-15,5.999e-15,5.004e-15,4.204e-15,3.571e-15,3.047e-15,2.597e-15,2.138e-15,
        1.956e-15,1.584e-15,1.227e-15,1.021e-15,8.356e-16,7.777e-16,6.812e-16,7.386e-16,6.128e-16,6.251e-16,
        5.519e-16,3.936e-16,4.141e-16,3.395e-16,3.002e-16,2.502e-16,2.273e-16,2.299e-16,1.429e-16,1.574e-16,
        1.218e-16,1.280e-16,1.612e-16,8.604e-17,9.270e-17,5.371e-17,5.495e-17,4.276e-17,3.693e-17,6.592e-17,
        6.261e-17,2.266e-17,3.924e-17,5.036e-17,3.051e-17,7.985e-17,1.630e-16,1.787e-16,5.729e-17,6.383e-18,
        5.257e-18,5.222e-18,4.369e-18,3.186e-18,3.915e-18,2.197e-18,1.690e-18,1.177e-18,9.963e-19,9.197e-19,
        6.790e-19,5.695e-19,5.234e-19,3.209e-19,2.809e-19,2.700e-19,1.624e-19,1.383e-19,1.192e-19,9.024e-20,
        9.442e-20,5.076e-20,6.390e-20,4.695e-20,2.734e-20,3.940e-20,2.067e-20,2.327e-20,2.294e-20,1.385e-20,
        1.932e-21,8.299e-21,5.854e-21,1.843e-21,0.,1.783e-21,4.490e-21,4.205e-21,*([0.0]*62)])

    nue_flux = np.array([
        1.530e-14, 5.722e-14, 1.273e-13, 1.231e-13, 1.042e-13, 1.078e-13, 1.093e-13, 1.086e-13, 1.061e-13, 1.034e-13,
        1.001e-13, 9.654e-14, 9.198e-14, 8.800e-14, 8.467e-14, 8.008e-14, 7.740e-14, 7.390e-14, 6.924e-14, 6.618e-14,
        6.239e-14, 6.037e-14, 5.633e-14, 5.446e-14, 5.014e-14, 4.838e-14, 4.520e-14, 4.350e-14, 4.028e-14, 3.933e-14,
        3.696e-14, 3.455e-14, 3.285e-14, 3.059e-14, 2.885e-14, 2.803e-14, 2.574e-14, 2.431e-14, 2.298e-14, 2.165e-14,
        2.042e-14, 1.867e-14, 1.763e-14, 1.656e-14, 1.545e-14, 1.485e-14, 1.361e-14, 1.281e-14, 1.190e-14, 1.130e-14,
        1.043e-14, 9.800e-15, 8.832e-15, 8.607e-15, 7.727e-15, 7.285e-15, 6.793e-15, 6.371e-15, 5.772e-15, 5.490e-15,
        4.989e-15, 4.656e-15, 4.211e-15, 4.071e-15, 3.819e-15, 3.496e-15, 3.165e-15, 2.922e-15, 2.624e-15, 2.489e-15,
        2.276e-15, 2.078e-15, 1.887e-15, 1.716e-15, 1.603e-15, 1.448e-15, 1.338e-15, 1.215e-15, 1.171e-15, 9.923e-16,
        9.308e-16, 8.357e-16, 7.638e-16, 6.755e-16, 6.545e-16, 5.973e-16, 5.257e-16, 4.645e-16, 4.304e-16, 3.828e-16,
        3.410e-16, 3.141e-16, 2.881e-16, 2.498e-16, 2.223e-16, 2.055e-16, 1.819e-16, 1.592e-16, 1.407e-16, 1.242e-16,
        1.142e-16, 1.028e-16, 8.425e-17, 7.409e-17, 6.574e-17, 5.592e-17, 4.790e-17, 4.200e-17, 3.153e-17, 2.980e-17,
        2.362e-17, 2.218e-17, 1.834e-17, 1.757e-17, 1.367e-17, 1.136e-17, 9.188e-18, 7.469e-18, 6.502e-18, 5.513e-18,
        4.571e-18, 4.365e-18, 2.147e-18, 2.322e-18, 1.548e-18, 1.282e-18, 1.049e-18, 8.226e-19, 8.297e-19, 6.143e-19,
        8.553e-19, 4.705e-19, 4.387e-19, 5.170e-19, 3.049e-19, 1.612e-19, 1.606e-19, 1.181e-19, 1.960e-19, 7.793e-20,
        1.571e-19, 1.169e-19, *([0.0] * 58)])

    nueb_flux = np.array([
        5.047e-15, 1.156e-14, 1.705e-14, 1.529e-14, 1.002e-14, 1.039e-14, 9.661e-15, 9.990e-15, 9.711e-15, 9.380e-15,
        9.049e-15, 9.298e-15, 8.340e-15, 8.007e-15, 7.769e-15, 7.364e-15, 6.980e-15, 6.944e-15, 6.564e-15, 5.783e-15,
        6.041e-15, 5.471e-15, 5.113e-15, 5.054e-15, 4.918e-15, 4.902e-15, 4.552e-15, 4.400e-15, 4.388e-15, 3.939e-15,
        3.598e-15, 3.530e-15, 3.588e-15, 3.289e-15, 3.112e-15, 2.919e-15, 2.733e-15, 2.850e-15, 2.564e-15, 2.514e-15,
        2.387e-15, 2.242e-15, 2.093e-15, 2.027e-15, 1.812e-15, 1.724e-15, 1.665e-15, 1.532e-15, 1.470e-15, 1.394e-15,
        1.305e-15, 1.279e-15, 1.140e-15, 1.042e-15, 9.795e-16, 9.706e-16, 8.481e-16, 8.082e-16, 7.190e-16, 6.964e-16,
        6.877e-16, 6.044e-16, 5.312e-16, 5.064e-16, 4.558e-16, 4.458e-16, 3.910e-16, 3.674e-16, 3.457e-16, 3.385e-16,
        2.937e-16, 2.900e-16, 2.523e-16, 2.363e-16, 2.107e-16, 1.936e-16, 1.801e-16, 1.561e-16, 1.465e-16, 1.371e-16,
        1.249e-16, 1.148e-16, 1.079e-16, 9.442e-17, 8.323e-17, 7.855e-17, 6.963e-17, 6.551e-17, 5.711e-17, 4.896e-17,
        4.631e-17, 4.348e-17, 3.697e-17, 3.522e-17, 3.216e-17, 2.985e-17, 2.577e-17, 2.299e-17, 2.077e-17, 1.950e-17,
        1.843e-17, 1.332e-17, 1.044e-17, 9.753e-18, 8.689e-18, 8.865e-18, 7.198e-18, 6.320e-18, 5.182e-18, 3.874e-18,
        3.370e-18, 3.068e-18, 2.157e-18, 2.255e-18, 1.785e-18, 1.269e-18, 1.316e-18, 8.751e-19, 8.883e-19, 6.479e-19,
        5.714e-19, 6.861e-19, 4.251e-19, 4.221e-19, 3.999e-19, 3.749e-19, 1.223e-19, 2.968e-19, 7.318e-20, 1.551e-20,
        3.392e-19, 2.261e-19, 1.258e-19, 9.574e-20, 8.005e-20, 4.006e-20, 1.595e-19, 0., 3.208e-19, 1.196e-19,
        1.993e-19, 1.595e-19, 3.987e-20, *([0.0] * 57)])

    # ── Observed energy bins (GeV) ──────────────────────────────────────────
    obs_Elo = np.array([475.,550.,675.,800.,950.,1100.,1300.,1500.]) / 1000.
    obs_Ehi = np.array([550.,675.,800.,950.,1100.,1300.,1500.,3000.]) / 1000.
    obs_centers = (obs_Elo + obs_Ehi) / 2.

    observed   = np.array([83., 90., 63., 58., 49., 45., 35., 67.])
    background = np.array([70.91, 81.51, 61.94, 58.95, 50.01, 44.49, 33.73, 65.27])

    covariance = np.array([
        [ 0.022008,  0.0012446, 0.0048977, 0.0034062, 0.0044717,-0.0028033, 0.00040001,0.0020426],
        [ 0.0012446, 0.02618,   0.0012558, 0.0050882, 0.00094575,0.0076589, 0.0036528, 0.011599 ],
        [ 0.0048977, 0.0012558, 0.026245,  0.013242,  0.015989,  0.013736,  0.018327,  0.016744 ],
        [ 0.0034062, 0.0050882, 0.013242,  0.034777,  0.023469,  0.024032,  0.033912,  0.026725 ],
        [ 0.0044717, 0.00094575,0.015989,  0.023469,  0.052856,  0.032156,  0.05459,   0.028499 ],
        [-0.0028033, 0.0076589, 0.013736,  0.024032,  0.032156,  0.082711,  0.069783,  0.050257 ],
        [ 0.00040001,0.0036528, 0.018327,  0.033912,  0.05459,   0.069783,  0.12506,   0.053062 ],
        [ 0.0020426, 0.011599,  0.016744,  0.026725,  0.028499,  0.050257,  0.053062,  0.064412 ],
    ])

    # ── Detector and beam parameters ────────────────────────────────────────
    N_target   = 2.81e32
    L_km       = 0.541
    pot        = 6.46e20
    sigma_e = np.full(len(E_centers), 1e-38)
    efficiency_e = np.full(len(E_centers), 0.20)

    sigma_mu = np.full(len(E_centers), 1e-38)
    
    
    eff_efc = np.array([
        0.        , 0.03515726, 0.1230604, 0.17592975, 0.2216032, 0.27385078,
        0.30083114, 0.30912699, 0.32330211, 0.32699299, 0.3247482, 0.32804201,
        0.32729473, 0.33095618, 0.31724309, 0.31813633, 0.30701209, 0.31377639,
        0.29874747, 0.29447149, 0.29277409, 0.28382266, 0.28116426, 0.27507559,
        0.27508131, 0.25247578])

    eff_epc = np.array([
        0.        , 0.00397274, 0.0160696, 0.02624025, 0.0370968, 0.05223922,
        0.07090886, 0.09739301, 0.11365789, 0.12735701, 0.1426418, 0.15891799,
        0.16618527, 0.17774382, 0.18275691, 0.19490367, 0.19950791, 0.21448361,
        0.21212253, 0.21639851, 0.22244591, 0.22269734, 0.22753574, 0.22927441,
        0.23578869, 0.24535422])

    eff_mufc = np.array([
        0.        , 0.        , 0.0527866,  0.24883974, 0.31415519, 0.32502141,
        0.33006293, 0.30783141, 0.27433627, 0.23660959, 0.20688314, 0.18432302,
        0.16461115, 0.15197599, 0.14028143, 0.13356655, 0.12439114, 0.11702154,
        0.11043099, 0.10520339, 0.10118324, 0.09604268, 0.09409804, 0.08869908,
        0.08551871, 0.05957836])

    eff_mupc = np.array([
        0.        , 0.        , 0.0201334,  0.09491026, 0.16292481, 0.21872859,
        0.28035707, 0.34216859, 0.40691373, 0.46339041, 0.50144686, 0.53442698,
        0.55413885, 0.57510401, 0.58888857, 0.60810345, 0.61310886, 0.62672846,
        0.63331901, 0.63229661, 0.63839676, 0.63728732, 0.65798196, 0.65505092,
        0.66865129, 0.67584164])


    crosssection_mu = np.array([
        0.06982607, 0.20947822, 0.34913037, 0.48878252, 0.56058346, 0.62124627,
        0.67516347, 0.71623957, 0.73788154, 0.75280018, 0.76258204, 0.76836158,
        0.76836158, 0.7703674,  0.77342192, 0.77669262, 0.78001502, 0.78333742,
        0.78512129, 0.78465474, 0.78418819, 0.78372164, 0.78325509, 0.78278854,
        0.78232199, 0.77975596])















    # ── True-to-reco response matrix ────────────────────────────────────────
    n_true, n_obs = len(E_centers), len(obs_centers)
    R = np.zeros((n_true, n_obs))
    for i, E in enumerate(E_centers):
        for j in range(n_obs):
            if obs_Elo[j] <= E < obs_Ehi[j]:
                R[i, j] = 1.; break
    rs = R.sum(axis=1, keepdims=True)
    R[rs[:,0] > 0] /= rs[rs[:,0] > 0]

    """改改改改改改"""

    response_matrix_e = R
    response_matrix_mu = R

    efficiency_matrix_e=np.diag(sigma_e * efficiency_e)
    efficiency_matrix_mu=np.diag(sigma_mu * efficiency_mu)

    return dict(
        E_true=E_centers, L_km=L_km,
        Elo=obs_Elo,
        Ehi=obs_Ehi,
        obs_centers=obs_centers,
        observed=observed,
        background=background,
        covariance=covariance,

        response_matrix_e  =response_matrix_e,
        response_matrix_eb =response_matrix_eb,
        response_matrix_mu =response_matrix_mu,
        response_matrix_mub=response_matrix_mub,

        efficiency_matrix_e  =efficiency_matrix_e,
        efficiency_matrix_eb =efficiency_matrix_eb,
        efficiency_matrix_mu =efficiency_matrix_mu,
        efficiency_matrix_mub=efficiency_matrix_mub,
########################################################################
        numu_flux=numu_flux * pot * N_target,
        numub_flux=numub_flux * pot * N_target,
        nue_flux=nue_flux * pot * N_target,
        nueb_flux=nueb_flux * pot * N_target,
    )
"""改改改"""


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 · Mixing matrix & oscillation probability
# ═══════════════════════════════════════════════════════════════════════════

def _rot(N, i, j, th, d=0.):
    """Return N×N complex rotation matrix in (i,j)-plane with CP phase d."""
    R = np.eye(N, dtype=complex)
    s, c = np.sin(th), np.cos(th)
    R[i,i], R[j,j] = c, c
    R[i,j] =  s * np.exp(-1j*d)
    R[j,i] = -s * np.exp( 1j*d)
    return R

def _U3(N):
    """Standard 3-flavor PMNS block embedded in N×N space."""
    return _rot(N,1,2,TH23) @ _rot(N,0,2,TH13,D13) @ _rot(N,0,1,TH12)

def build_mixing_matrix(model, p):
    """
    Construct PMNS mixing matrix U and mass-squared array m2.

    Parameters
    ----------
    model : '3' | '3+1' | '3+2' | '1+3+1'
    p     : dict  –  keys are sin(angle) values (not angles), dm41, dm51, delta phases

    Returns
    -------
    U  : complex ndarray (N×N)
    m2 : float  ndarray (N,)
    """
    sa = lambda v: np.arcsin(np.clip(v, 0., 1.))   # sin → angle

    if model == '3':
        return _U3(3), M2_STD.copy()

    if model == '3+1':
        N = 4
        Us = (_rot(N,2,3,sa(p.get('theta34',0.)),p.get('delta34',0.)) @
              _rot(N,1,3,sa(p.get('theta24',0.)),p.get('delta24',0.)) @
              _rot(N,0,3,sa(p.get('theta14',0.)),p.get('delta14',0.)))
        return Us @ _U3(N), np.append(M2_STD, p.get('dm41', 1.0))

    if model in ('3+2', '1+3+1'):
        N = 5
        d41_def, d51_def = (-0.87, 0.47) if model == '1+3+1' else (0.47, 0.87)
        Us1 = (_rot(N,2,3,sa(p.get('theta34',0.)),p.get('delta34',0.)) @
               _rot(N,1,3,sa(p.get('theta24',0.)),p.get('delta24',0.)) @
               _rot(N,0,3,sa(p.get('theta14',0.)),p.get('delta14',0.)))
        Us2 = (_rot(N,3,4,sa(p.get('theta45',0.)),p.get('delta45',0.)) @
               _rot(N,2,4,sa(p.get('theta35',0.)),p.get('delta35',0.)) @
               _rot(N,1,4,sa(p.get('theta25',0.)),p.get('delta25',0.)) @
               _rot(N,0,4,sa(p.get('theta15',0.)),p.get('delta15',0.)))
        return Us2 @ Us1 @ _U3(N), np.append(M2_STD, [p.get('dm41',d41_def), p.get('dm51',d51_def)])

    raise ValueError(f"Unknown model '{model}'. Choose '3', '3+1', '3+2', or '1+3+1'.")

# ═══════════════════════════════════════════════════════════════════════════
# Part 3 · Event rate (spectrum) calculation
# ═══════════════════════════════════════════════════════════════════════════
def osc_prob(U, m2, alpha, beta, L, E_arr, anti=False):

    phase  = np.exp(-1j * 1.267 * m2[None, :] * L / E_arr[:, None])
    coeffs = np.conj(U[beta,:]) * U[alpha,:] if anti else U[beta,:] * np.conj(U[alpha,:])
    return np.abs(phase @ coeffs) ** 2
"""question 1+3+1 / 3+2"""


def event_calculation(params, exp):

    U, m2 = build_mixing_matrix(MODEL, params)
    E, L  = exp['E_true'], exp['L_km']

    P_mu_e   = osc_prob(U, m2, 1, 0, L, E, anti=False)
    P_mub_e  = osc_prob(U, m2, 1, 0, L, E, anti=True)
    P_e_e    = osc_prob(U, m2, 0, 0, L, E, anti=False)
    P_eb_e   = osc_prob(U, m2, 0, 0, L, E, anti=True)
    P_e_mu   = osc_prob(U, m2, 0, 1, L, E, anti=False)
    P_eb_mu  = osc_prob(U, m2, 0, 1, L, E, anti=True)
    P_mu_mu  = osc_prob(U, m2, 1, 1, L, E, anti=False)
    P_mub_mu = osc_prob(U, m2, 1, 1, L, E, anti=True)

    signalnu_e  = exp['numu_flux'] * P_mu_e + exp['numu_fluxb'] * P_mub_e
    signalnu_mu = exp['nue_flux'] * P_e_mu + exp['nue_fluxb'] * P_eb_mu
    bkgnu_e = exp['nue_flux'] * P_e_e + exp['nue_fluxb'] * P_eb_e
    bkgnu_mu = exp['numu_flux'] * P_mu_mu + exp['numu_fluxb'] * P_mub_mu

    response_matrix_e = exp['response_matrix_e']
    response_matrix_mu = exp['response_matrix_mu']

    efficiency_matrix_e = exp['efficiency_matrix_e']
    efficiency_matrix_mu = exp['efficiency_matrix_mu']

    event_signalnu_e  = response_matrix_e  @ efficiency_matrix_e  @ signalnu_e
    event_signalnu_mu = response_matrix_mu @ efficiency_matrix_mu @ signalnu_mu
    event_bkgnu_e     = response_matrix_e  @ efficiency_matrix_e  @ bkgnu_e
    event_bkgnu_mu    = response_matrix_mu @ efficiency_matrix_mu @ bkgnu_mu

    return (
        signalnu_e, signalnu_mu,
        bkgnu_e, bkgnu_mu,
        event_signalnu_e, event_signalnu_mu,
        event_bkgnu_e,event_bkgnu_mu
    )


def compute_spectrum(params, exp):
    (signalnu_e, signalnu_mu, bkgnu_e, bkgnu_mu,
     event_signalnu_e, event_signalnu_mu,
     event_bkgnu_e, event_bkgnu_mu) = event_calculation(params, exp)

    """合并bin"""
    """合并bin"""
    """合并bin"""

    return events



# ═══════════════════════════════════════════════════════════════════════════
# Part 4 · Chi-squared, global fit, and 2D scans
# ═══════════════════════════════════════════════════════════════════════════

# ── Optimizer parameter-space helpers ─────────────────────────────────────────

def _to_opt(val, nm):
    return np.log10(max(float(val), 1e-12)) if PARAM_CFG[nm]['sc'] == 'log' else float(val)

def _from_opt(val, nm):
    return 10.**float(val) if PARAM_CFG[nm]['sc'] == 'log' else float(val)

def _x2p(x):
    """Optimizer vector → physical param dict."""
    return {nm: _from_opt(x[i], nm) for i, nm in enumerate(PARAM_NAMES)}

def _p2x(p):
    """Physical param dict → optimizer vector (init fallback for missing keys)."""
    return np.array([_to_opt(p.get(nm, PARAM_CFG[nm]['init']), nm) for nm in PARAM_NAMES])

def _get_bounds():
    """L-BFGS-B bounds in optimizer space for all PARAM_NAMES."""
    out = []
    for nm in PARAM_NAMES:
        lo, hi = PARAM_CFG[nm]['lo'], PARAM_CFG[nm]['hi']
        if PARAM_CFG[nm]['sc'] == 'log':
            lo, hi = np.log10(max(lo, 1e-12)), np.log10(max(hi, 1e-12))
        out.append((lo, hi))
    return out

# ── Covariance helpers ────────────────────────────────────────────────────────

def _abs_cov_inv(frac_cov, mu):
    """Absolute cov = frac_cov × outer(μ,μ), regularised and inverted."""
    C = frac_cov * np.outer(mu, mu)
    C += 1e-8 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    try:    return np.linalg.inv(C)
    except: return np.linalg.pinv(C)

def chi2_value(params, exp):
    """Chi-squared for params relative to MiniBooNE data."""

    """卡方计算核心"""
    """卡方计算核心"""
    """卡方计算核心"""


    mu    = np.maximum(compute_spectrum(params, exp) + exp['background'], 1e-12)
    inv_c = _abs_cov_inv(exp['covariance'], mu)
    d     = exp['observed'] - mu
    return float(d @ inv_c @ d)

def chi2_null(exp):
    """Chi-squared for the background-only (no oscillation) hypothesis."""
    bkg   = exp['background']
    inv_c = _abs_cov_inv(exp['covariance'], bkg)
    d     = exp['observed'] - bkg
    return float(d @ inv_c @ d)


# ── Global minimiser ──────────────────────────────────────────────────────────

def global_minimize(exp, n_random=50, seed=42):

    bounds = _get_bounds()
    x0     = _p2x({nm: PARAM_CFG[nm]['init'] for nm in PARAM_NAMES})
    rng    = np.random.default_rng(seed)
    obj    = lambda x: chi2_value(_x2p(x), exp) / 2.   # minimise chi2/2 = −log L

    blo = [lo for lo,_ in bounds]; bhi = [hi for _,hi in bounds]

    # Build multi-start pool: init, perturbed, theta-targeted, random
    starts = [x0]
    for _ in range(15):
        starts.append(np.clip(x0 + rng.normal(0., 0.4, x0.shape), blo, bhi))
    for _ in range(5):
        pt = x0.copy()
        for k, nm in enumerate(PARAM_NAMES):
            if nm.startswith('theta') and PARAM_CFG[nm]['hi'] > 0.01:
                pt[k] = (rng.uniform(np.log10(0.05), np.log10(0.5))
                         if PARAM_CFG[nm]['sc'] == 'log'
                         else rng.uniform(0.05, 0.5))
        starts.append(np.clip(pt, blo, bhi))
    for _ in range(n_random):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

    opt = dict(method='L-BFGS-B', bounds=bounds,
               options={'maxiter':1000,'ftol':1e-10,'gtol':1e-9})
    best_x, best_c2 = None, np.inf
    for xs in starts:
        try:
            res = minimize(obj, xs, **opt)
            c2  = 2. * res.fun
            if c2 < best_c2:
                best_c2, best_x = c2, res.x.copy()
        except Exception: continue

    # Final fine refinement
    try:
        res2 = minimize(obj, best_x, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter':5000,'ftol':1e-14,'gtol':1e-12})
        if 2.*res2.fun < best_c2:
            best_c2, best_x = 2.*res2.fun, res2.x.copy()
    except Exception: pass

    return _x2p(best_x), float(best_c2)


# ── Fixed 2D scan ─────────────────────────────────────────────────────────────

def scan_fixed_2d(ax1, grid1, ax2, grid2, exp,
                  extra_fixed=None, couple=None):

    base = {nm: PARAM_CFG[nm]['init'] for nm in PARAM_NAMES}
    if extra_fixed:
        base.update({k: v for k, v in extra_fixed.items() if k in PARAM_NAMES})

    n1, n2 = len(grid1), len(grid2)
    chi2_grid   = np.full((n1, n2), np.nan)
    param_store = [[None]*n2 for _ in range(n1)]

    for i, v1 in enumerate(grid1):
        for j, v2 in enumerate(grid2):
            p = dict(base)
            if ax1 in PARAM_NAMES: p[ax1] = float(v1)
            if ax2 in PARAM_NAMES: p[ax2] = float(v2)
            if couple: couple(p, v1, v2)
            chi2_grid[i,j]    = chi2_value(p, exp)
            param_store[i][j] = dict(p)

    chi2_min = float(np.nanmin(chi2_grid))
    TS_grid  = np.maximum(0., chi2_grid - chi2_min)
    idx      = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
    return (chi2_grid, TS_grid,
            (float(grid1[idx[0]]), float(grid2[idx[1]])),
            chi2_min, param_store[idx[0]][idx[1]])












### profile

































# ═══════════════════════════════════════════════════════════════════════════
# Part 5 · Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_spectrum(exp, best_params, savepath=None):
    """
    绘制能谱图：
    1. 使用相对协方差矩阵计算绝对误差，并应用到观测数据的误差棒上。
    2. 移除了预估数据的颜色填充阴影。
    """
    bkg   = exp['background']
    sig   = compute_spectrum(best_params, exp)
    total = bkg + sig  # 理论预测值 N_i
    data  = exp['observed']
    E     = exp['obs_centers']
    xerr  = [E - exp['Elo'], exp['Ehi'] - E]

    # ── 核心逻辑：从相对协方差计算绝对误差 (sigma_i) ────────────────────────
    
    frac_cov = exp['covariance']
    # 计算绝对误差 sigma_i = sqrt(相对方差_ii) * 预测值_i
    # np.diag(frac_cov) 提取对角线上的相对方差
    total_err = np.sqrt(np.diag(frac_cov)) * total
    
    # 准备阶梯图数据 (用于预测线)
    E_step = np.append(exp['Elo'], exp['Ehi'][-1])
    T_step = np.append(total, total[-1])
    B_step = np.append(bkg, bkg[-1])

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. 绘制背景线 (虚线)
    ax.step(E_step, B_step, where='post',
            color='steelblue', ls='--', lw=1.6, label='Background only')
    
    # 2. 绘制总预测线 (实线)
    ax.step(E_step, T_step, where='post',
            color='steelblue', ls='-', lw=2., label='Bkg + prediction (signal)')

    # 【此处已完全删去 ax.fill_between 颜色填充代码】

    # 3. 绘制观测数据点 (使用计算出的真实误差 total_err)
    ax.errorbar(E, data, 
                yerr=total_err,           # <--- 关键修改：使用协方差导出的真实误差
                xerr=xerr, 
                fmt='o', 
                color='crimson', 
                ms=7, 
                capsize=4, 
                elinewidth=1.5,
                markeredgewidth=1.2, 
                label='Observed data (w/ Covariance Error)')

    # ── 图形修饰 ────────────────────────────────────────────────────────────
    ax.set_xlabel(r'Reconstructed $E_\nu^{\rm QE}\ \rm [GeV]$', fontsize=13)
    ax.set_ylabel('Event count', fontsize=13)
    ax.set_title(f'MicroBooNE — Prediction vs Data', fontsize=12)
    
    ax.legend(fontsize=11, loc='upper right', framealpha=0.85)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=11)
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_xlim(exp['Elo'][0]*0.98, exp['Ehi'][-1]*1.02)
    ax.set_ylim(bottom=0.)
    
    fig.tight_layout()
    if savepath: fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, ax


def plot_contour(grid1, grid2, TS_grid, xlabel, ylabel, title,
                 xscale='log', yscale='log',
                 best_xy=None, global_best_xy=None, ts_at_global=None,
                 chi2_null_val=None, chi2_scan_min=None, chi2_global_min=None,
                 savepath=None):
    """Two-panel plot: (left) chi2 heatmap, (right) allowed contours."""
    crits  = [chi2_dist.ppf(cl, df=2) for cl in [0.68, 0.90, 0.99]]
    labels = ['68% CL', '90% CL', '99% CL']
    styles = [':', '-', '--']
    alphas = [0.15, 0.10, 0.05]
    ts_max = float(np.nanmax(TS_grid))
    X, Y   = np.meshgrid(grid1, grid2, indexing='ij')

    # ── Figure layout ────────────────────────────────────────────────────────
    sq, cb_gap, cb_w = 5.5, 0.10, 0.22
    pad_l, pad_r, pad_t, pad_b, gap = 0.85, 0.35, 0.75, 0.80, 1.5
    fw = pad_l + sq + cb_gap + cb_w + gap + sq + pad_r
    fh = pad_t + sq + pad_b
    fig = plt.figure(figsize=(fw, fh))
    ax1  = fig.add_axes([pad_l/fw,                      pad_b/fh, sq/fw,    sq/fh])
    cax  = fig.add_axes([(pad_l+sq+cb_gap)/fw,          pad_b/fh, cb_w/fw,  sq/fh])
    ax2  = fig.add_axes([(pad_l+sq+cb_gap+cb_w+gap)/fw, pad_b/fh, sq/fw,    sq/fh])

    # ── Left: heatmap ────────────────────────────────────────────────────────
    vmax = min(ts_max*1.1, crits[-1]*4.) if ts_max > 0 else 10.
    pcm  = ax1.pcolormesh(X, Y, TS_grid, shading='auto', cmap='viridis_r', vmin=0., vmax=vmax)
    cb   = fig.colorbar(pcm, cax=cax)
    cb.set_label(r'$\Delta\chi^2$', fontsize=14, labelpad=8)
    cb.ax.tick_params(labelsize=12)
    for level, ls in zip(crits, styles):
        if ts_max >= level:
            ax1.contour(X, Y, TS_grid, levels=[level], colors=['r'],
                        linestyles=[ls], linewidths=1.8, zorder=3)
    if best_xy:
        ax1.plot(*best_xy, '*', color='gold', ms=14, mec='k', mew=0.8, zorder=7,
                 label=f'Best fit (TS≡0)\n({best_xy[0]:.3g}, {best_xy[1]:.3g})')
        ax1.legend(fontsize=10, loc='upper right', framealpha=0.85)

    # ── Right: allowed regions ───────────────────────────────────────────────
    legend_h = []
    for level, lbl, ls, fa in zip(crits, labels, styles, alphas):
        if ts_max >= level:
            try:
                ax2.contour(X, Y, TS_grid, levels=[level], colors=['r'], linestyles=[ls], linewidths=2.)
                ax2.contourf(X, Y, TS_grid, levels=[0., level], colors=['r'], alpha=fa)
            except Exception: pass
        legend_h.append(Line2D([0],[0], color='r' if ts_max >= level else '#aaa',
                                ls=ls, lw=2., label=lbl))
    if best_xy:
        pt, = ax2.plot(*best_xy, '*', color='gold', ms=14, mec='k', mew=0.8,
                       zorder=7, label='Best fit (TS≡0)')
        legend_h.append(pt)

    show_global = (global_best_xy is not None and ts_at_global is not None
                   and np.isfinite(ts_at_global)
                   and ts_at_global < chi2_dist.ppf(0.90, 2))
    if show_global:
        gpt, = ax2.plot(*global_best_xy, 'D', color='cyan', ms=9, mec='k', mew=0.8, zorder=6,
                        label=f'Global best proj. (TS={ts_at_global:.2f})')
        legend_h.append(gpt)

    info = []
    if chi2_null_val    is not None: info.append(r'$\chi^2_{\rm null}='      + f'{chi2_null_val:.2f}$')
    if chi2_scan_min    is not None: info.append(r'$\chi^2_{\rm scan\,min}=' + f'{chi2_scan_min:.2f}$')
    if chi2_global_min  is not None: info.append(r'$\chi^2_{\rm global}='    + f'{chi2_global_min:.2f}$')
    if chi2_null_val and chi2_scan_min:
        info.append(r'$\Delta\chi^2_{\rm sig}=' + f'{chi2_null_val - chi2_scan_min:.2f}$')
    if ts_at_global and not show_global and np.isfinite(ts_at_global):
        info.append(r'Global proj. $\Delta{\rm TS}=' + f'{ts_at_global:.2f}$')
    if info:
        ax2.text(0.03, 0.03, '\n'.join(info), transform=ax2.transAxes,
                 fontsize=10, ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='gray', alpha=0.82))
    ax2.legend(handles=legend_h, loc='upper right', fontsize=10, framealpha=0.85)

    for ax in (ax1, ax2):
        ax.set_xscale(xscale); ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=14); ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
        ax.grid(True, which='both', ls='--', alpha=0.25)
    ax1.set_title(f"MiniBooNE {MODEL} — {title}  " + r"$\Delta\chi^2$ map", fontsize=12, pad=8)
    ax2.set_title(f"MiniBooNE {MODEL} — {title}  Allowed regions",           fontsize=12, pad=8)

    if savepath: fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()

load_experiment

