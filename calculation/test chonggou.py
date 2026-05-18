"""
MiniBooNE 1+3+1 Sterile Neutrino Oscillation Analysis  ·  Refactored
======================================================================
Structure
---------
  Part 0 – Imports & global config  (MODEL, SCAN_MODE, PARAM_CFG, SCAN_TASKS)
  Part 1 – load_experiment()         → exp  (dict with all detector/beam data)
  Part 2 – build_mixing_matrix()     → U, m2
            osc_prob()               → oscillation probability array
  Part 3 – compute_spectrum()        → predicted signal counts per obs bin
  Part 4 – chi2_value() / chi2_null()
            global_minimize()        → best_params, chi2_global_min
            scan_fixed_2d()          → chi2_grid, TS_grid, best_xy, chi2_min, best_fp
            scan_profile_2d()        → chi2_grid, TS_grid, best_xy, chi2_global_min, best_fp
  Part 5 – plot_spectrum() / plot_contour()
  Part 6 – main()

Usage
-----
  python miniboone_refactored.py

  Toggle global settings below:
    MODEL      – oscillation model ('3', '3+1', '1+3+1')
    SCAN_MODE  – 'fixed' (fast) or 'profile' (slower, marginalises nuisances)
    RUN_GLOBAL – compute full global best-fit before scanning
"""

# ═══════════════════════════════════════════════════════════════════════════
# Part 0 · Imports & global configuration
# ═══════════════════════════════════════════════════════════════════════════
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

# ── Parameter names for 1+3+1 / 3+2 (5-neutrino) models ────────────────────
PARAM_NAMES = [
    'theta14','theta24','theta34','theta15','theta25','theta35','theta45',
    'delta14','delta24','delta34','delta15','delta25','delta35','delta45',
    'dm41','dm51',
]

# ── Parameter config: initial value, optimizer bounds, log/linear scale ───────
_z = dict(init=0., lo=0., hi=0., sc='linear')   # 复制了一份 _z 字典，从而快速将它们全部设为固定零值
PARAM_CFG = {
    'dm41'   : dict(init=0.87, lo=0.01, hi=100.,         sc='log'),
    'dm51'   : dict(init=0.47, lo=0.01, hi=100.,         sc='log'),
    'theta14': dict(init=0.15, lo=1e-4, hi=1.0, sc='log'),
    'theta15': dict(init=0.13, lo=1e-4, hi=1.0, sc='log'),
    'theta24': dict(init=0.13, lo=1e-4, hi=1.0, sc='log'),
    'theta25': dict(init=0.17, lo=1e-4, hi=1.0, sc='log'),
    **{k: dict(**_z) for k in ['theta34','theta35','theta45',
                                'delta14','delta15','delta24','delta25',
                                'delta34','delta35','delta45']},
}

# ── Coupling functions: map virtual scan axes to real parameters ─────────────
def _couple_mode1(p, v1, v2):
    """sin(th14)*sin(th24) and sin(th15)*sin(th25) axes."""
    p['theta14'] = p['theta24'] = float(np.clip(np.sqrt(max(v1, 0.)), 0., 1.))
    p['theta15'] = p['theta25'] = float(np.clip(np.sqrt(max(v2, 0.)), 0., 1.))

def _couple_mode3(p, v1, v2):
    """sin²(2θ_eff) = 4 sin²θ14 sin²θ24 axis."""
    st = float(np.clip((np.clip(v1, 1e-9, 1.) / 4.) ** 0.25, 0., 1.))
    p['theta14'] = p['theta24'] = st
    p['dm41'] = float(v2)

_zero_roles = {k: 0. for k in ['theta34','theta35','theta45',
                                 'delta14','delta15','delta24','delta25',
                                 'delta34','delta35','delta45']}

# ── Scan task definitions ─────────────────────────────────────────────────────
SCAN_TASKS = [
    dict(
        ax1='_prod14',  ax2='_prod15',
        grid1=np.logspace(np.log10(1e-4),        np.log10(np.sqrt(0.9)), 50),
        grid2=np.logspace(np.log10(1e-4),        np.log10(np.sqrt(0.9)), 50),
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
]


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
    sigma_flux = np.full(len(E_centers), 1e-38)
    efficiency = np.full(len(E_centers), 0.20)

    # ── True-to-reco response matrix ────────────────────────────────────────
    n_true, n_obs = len(E_centers), len(obs_centers)
    R = np.zeros((n_true, n_obs))
    for i, E in enumerate(E_centers):
        for j in range(n_obs):
            if obs_Elo[j] <= E < obs_Ehi[j]:
                R[i, j] = 1.; break
    rs = R.sum(axis=1, keepdims=True)
    R[rs[:,0] > 0] /= rs[rs[:,0] > 0]

    # ── Pre-computed flux × efficiency × POT × N_target weights ─────────────
    scale = sigma_flux * efficiency * pot * N_target

    return dict(
        E_true=E_centers, L_km=L_km,
        Elo=obs_Elo, Ehi=obs_Ehi, obs_centers=obs_centers,
        observed=observed, background=background, covariance=covariance,
        response_matrix=R,
        precomp_nu =numu_flux  * scale,
        precomp_nub=numub_flux * scale,
    )


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


def osc_prob(U, m2, alpha, beta, L, E_arr, anti=False):
    """
    Oscillation probability P(ν_α → ν_β) for an array of energies.

    Uses the standard formula with phase = 1.267 * Δm² * L / E.
    """
    phase  = np.exp(-1j * 1.267 * m2[None, :] * L / E_arr[:, None])
    coeffs = np.conj(U[beta,:]) * U[alpha,:] if anti else U[beta,:] * np.conj(U[alpha,:])
    return np.abs(phase @ coeffs) ** 2


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 · Event rate (spectrum) calculation
# ═══════════════════════════════════════════════════════════════════════════

def compute_spectrum(params, exp):
    """
    Predicted νe-appearance signal in each observed energy bin.

    Computes numu→nue oscillation for both neutrinos and antineutrinos,
    weights by pre-computed flux×cross-section×efficiency×POT×N_target,
    then smears via the response matrix.

    Returns
    -------
    signal : float ndarray, shape (n_obs_bins,)
    """
    U, m2 = build_mixing_matrix(MODEL, params)
    E, L  = exp['E_true'], exp['L_km']
    P_nu  = osc_prob(U, m2, 1, 0, L, E, anti=False)
    P_nub = osc_prob(U, m2, 1, 0, L, E, anti=True)
    raw   = exp['precomp_nu'] * P_nu + exp['precomp_nub'] * P_nub
    return exp['response_matrix'].T @ raw


# ═══════════════════════════════════════════════════════════════════════════
# Part 4 · Chi-squared, global fit, and 2D scans
# ═══════════════════════════════════════════════════════════════════════════

# ── Covariance helpers ────────────────────────────────────────────────────────

def _abs_cov_inv(frac_cov, mu):
    """Absolute cov = frac_cov × outer(μ,μ), regularised and inverted."""
    C = frac_cov * np.outer(mu, mu)
    C += 1e-8 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    try:    return np.linalg.inv(C)
    except: return np.linalg.pinv(C)

def chi2_value(params, exp):
    """Chi-squared for params relative to MiniBooNE data."""
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


# ── Global minimiser ──────────────────────────────────────────────────────────

def global_minimize(exp, n_random=50, seed=42):
    """
    Multi-start L-BFGS-B global minimisation of chi2.

    Returns
    -------
    best_params : dict  –  physical parameter values at minimum
    chi2_min    : float –  minimum chi-squared value
    """
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
    """
    2D chi-squared grid scan with all parameters (except scan axes) fixed at
    their initial values (or extra_fixed overrides).

    Returns
    -------
    chi2_grid : (n1,n2) array of chi2 values
    TS_grid   : (n1,n2) array of ΔTS = chi2 − chi2_min  (≥0)
    best_xy   : (v1, v2) at minimum
    chi2_min  : float
    best_fp   : param dict at minimum
    """
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


# ── Profile likelihood helpers ────────────────────────────────────────────────

def _build_fixed_dict(ax1, ax2, v1, v2, param_roles, couple):
    """
    Construct the dict of all non-free parameters for one (v1,v2) grid point.
    Virtual axes (names starting with '_') are handled by the couple function.
    """
    # Start with scan-axis values (couple may overwrite via virtual names)
    scan_dict = {ax1: float(v1), ax2: float(v2)}
    if couple: couple(scan_dict, v1, v2)

    fixed = {}
    for nm in PARAM_NAMES:
        if nm in scan_dict:
            fixed[nm] = float(scan_dict[nm])   # set by direct axis or couple
            continue
        role = param_roles.get(nm, 'fixed_init')
        if   role == 'free':       pass                              # will be optimised
        elif role == 'fixed_init': fixed[nm] = PARAM_CFG[nm]['init']
        else:                      fixed[nm] = float(role)          # numeric constant
    return fixed

def _minimize_free(fixed, free_nms, free_bds, exp, global_best_p=None, seed=0):
    """
    Minimise chi2 over free_nms, with all other params held in fixed.

    Returns (best_param_dict, chi2_min).
    """
    rng = np.random.default_rng(seed)

    def obj(free_x):
        p = dict(fixed)
        for k, nm in enumerate(free_nms):
            p[nm] = _from_opt(free_x[k], nm)
        return chi2_value(p, exp) / 2.

    x0_free = np.array([np.clip(_to_opt(PARAM_CFG[nm]['init'], nm), lo, hi)
                        for nm, (lo, hi) in zip(free_nms, free_bds)])
    starts = [x0_free]
    if global_best_p is not None:
        gx = np.array([np.clip(_to_opt(global_best_p.get(nm, PARAM_CFG[nm]['init']), nm), lo, hi)
                       for nm, (lo, hi) in zip(free_nms, free_bds)])
        starts.insert(0, gx)
    for _ in range(5):
        starts.append(np.clip(x0_free + rng.normal(0., 0.3, x0_free.shape),
                               [lo for lo,_ in free_bds], [hi for _,hi in free_bds]))
    for _ in range(5):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in free_bds]))

    opt = dict(method='L-BFGS-B', bounds=free_bds,
               options={'maxiter':2000,'ftol':1e-10,'gtol':1e-9})
    best_fun, best_x = np.inf, starts[0]
    for xs in starts:
        try:
            res = minimize(obj, xs, **opt)
            if res.fun < best_fun:
                best_fun, best_x = res.fun, res.x.copy()
        except Exception: continue
    try:
        res2 = minimize(obj, best_x, method='L-BFGS-B', bounds=free_bds,
                        options={'maxiter':5000,'ftol':1e-14,'gtol':1e-12})
        if res2.fun < best_fun:
            best_fun, best_x = res2.fun, res2.x.copy()
    except Exception: pass

    best_p = dict(fixed)
    for k, nm in enumerate(free_nms):
        best_p[nm] = _from_opt(best_x[k], nm)
    return best_p, float(2. * best_fun)



# ── TS at global best projection ──────────────────────────────────────────────

def ts_at_projection(grid1, grid2, chi2_grid, chi2_ref, proj_xy,
                     xscale='log', yscale='log'):
    """Return ΔTS at the grid cell closest to proj_xy."""
    px, py = proj_xy
    if not (px and py and np.isfinite(px) and np.isfinite(py) and px > 0 and py > 0):
        return np.inf
    tol = 1.05
    for val, g, sc in [(px, grid1, xscale), (py, grid2, yscale)]:
        rng = g[-1] - g[0]
        if sc == 'log':
            if val < g[0]/tol or val > g[-1]*tol: return np.inf
        else:
            if val < g[0]-0.05*rng or val > g[-1]+0.05*rng: return np.inf
    di = (int(np.argmin(np.abs(np.log10(grid1)-np.log10(px))))
          if xscale == 'log' else int(np.argmin(np.abs(grid1-px))))
    dj = (int(np.argmin(np.abs(np.log10(grid2)-np.log10(py))))
          if yscale == 'log' else int(np.argmin(np.abs(grid2-py))))
    c2 = chi2_grid[di, dj]
    return np.inf if np.isnan(c2) else float(max(0., c2 - chi2_ref))


# ═══════════════════════════════════════════════════════════════════════════
# Part 5 · Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_spectrum(exp, best_params, savepath=None):
    """Plot background + predicted signal vs observed data."""
    bkg   = exp['background']
    sig   = compute_spectrum(best_params, exp)
    total = bkg + sig
    data  = exp['observed']
    E     = exp['obs_centers']
    xerr  = [E - exp['Elo'], exp['Ehi'] - E]

    E_step = np.append(exp['Elo'], exp['Ehi'][-1])
    T_step = np.append(total, total[-1])
    Te_step = np.append(np.sqrt(np.maximum(total, 1.)), np.sqrt(np.maximum(total[-1], 1.)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(E_step, np.append(bkg, bkg[-1]), where='post',
            color='steelblue', ls='--', lw=1.6, label='Background only')
    ax.step(E_step, T_step, where='post',
            color='steelblue', ls='-', lw=2., label='Bkg + prediction (signal)')
    ax.fill_between(E_step, T_step - Te_step, T_step + Te_step,
                    step='post', color='steelblue', alpha=0.18, label=r'Prediction $\pm\sqrt{N}$')
    ax.errorbar(E, data, yerr=np.sqrt(np.maximum(data, 1.)), xerr=xerr,
                fmt='o', color='crimson', ms=7, capsize=4, elinewidth=1.5,
                markeredgewidth=1.2, label='Observed data')

    ax.text(0.97, 0.97,
            f"Total bkg    = {bkg.sum():.1f}\n"
            f"Total signal = {sig.sum():.2f}\n"
            f"Total pred   = {total.sum():.1f}\n"
            f"Total data   = {data.sum():.0f}",
            transform=ax.transAxes, fontsize=11, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.85))

    ax.set_xlabel(r'Reconstructed $E_\nu^{\rm QE}\ \rm [GeV]$', fontsize=13)
    ax.set_ylabel('Event count', fontsize=13)
    ax.set_title(f'MiniBooNE {MODEL} — Bkg + Prediction vs Data', fontsize=12)
    ax.legend(fontsize=11, loc='lower left', framealpha=0.85)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=11)
    ax.grid(True, ls='--', alpha=0.35)
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


# ═══════════════════════════════════════════════════════════════════════════
# Part 6 · Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── Load data ────────────────────────────────────────────────────────────
    exp = load_experiment()

    # ── Null hypothesis ──────────────────────────────────────────────────────
    c2_null = chi2_null(exp)
    print(f"\nchi2_null = {c2_null:.4f}")

    # ── Global best-fit ──────────────────────────────────────────────────────
    best_params, c2_global = None, None
    if RUN_GLOBAL or SCAN_MODE == 'profile':
        print("\nRunning global_minimize (n_random=50) …")
        best_params, c2_global = global_minimize(exp, n_random=50, seed=42)
        delta = c2_null - c2_global
        print("=" * 60)
        print(f"  chi2_null       = {c2_null:.4f}")
        print(f"  chi2_global_min = {c2_global:.4f}")
        print(f"  Delta_chi2      = {delta:.4f}  ({np.sqrt(max(delta,0.)):.2f} sigma)")
        print("\n  Best-fit parameters:")
        for nm in ['theta14','theta24','theta15','theta25','dm41','dm51']:
            print(f"    {nm:12s} = {best_params.get(nm, 0.):.6g}")

        plot_spectrum(exp, best_params, savepath='miniboone_spectrum_bestfit.png')

    # ── 2D parameter scans ───────────────────────────────────────────────────
    print(f"\nScan method: {SCAN_MODE.upper()}")
    print("=" * 60)

    for k, task in enumerate(SCAN_TASKS):
        print(f"\n[Task {k+1}] {task['ax1']} vs {task['ax2']}"
              f"  ({len(task['grid1'])}×{len(task['grid2'])})")

        if SCAN_MODE == 'fixed':
            chi2_grid, TS_grid, best_xy, c2_min, best_fp = scan_fixed_2d(
                task['ax1'], task['grid1'], task['ax2'], task['grid2'], exp,
                extra_fixed=task.get('extra_fixed'), couple=task.get('couple'))
            chi2_ref = c2_min

        elif SCAN_MODE == 'profile':
            chi2_grid, TS_grid, best_xy, chi2_ref, best_fp = scan_profile_2d(
                task['ax1'], task['grid1'], task['ax2'], task['grid2'], exp,
                param_roles=task.get('param_roles'), couple=task.get('couple'),
                chi2_global_min=c2_global, global_best_p=best_params)
            c2_min = float(np.nanmin(chi2_grid))

        else:
            raise ValueError(f"Unknown SCAN_MODE='{SCAN_MODE}'. Use 'fixed' or 'profile'.")

        print(f"  chi2_scan_min = {c2_min:.4f}  "
              f"Δchi2_sig = {c2_null - c2_min:.4f}  "
              f"best = ({best_xy[0]:.4g}, {best_xy[1]:.4g})")

        # Project global best-fit onto scan axes
        global_xy = ts_gb = None
        if best_params is not None and task.get('proj_fn'):
            global_xy = task['proj_fn'](best_params)
            ts_gb = ts_at_projection(task['grid1'], task['grid2'], chi2_grid, chi2_ref,
                                     global_xy, task['xscale'], task['yscale'])

        plot_contour(
            task['grid1'], task['grid2'], TS_grid,
            task['xlabel'], task['ylabel'], task['title'],
            task['xscale'], task['yscale'],
            best_xy=best_xy, global_best_xy=global_xy, ts_at_global=ts_gb,
            chi2_null_val=c2_null, chi2_scan_min=c2_min, chi2_global_min=c2_global,
            savepath=task.get('savepath'),
        )


if __name__ == '__main__':
    main()






