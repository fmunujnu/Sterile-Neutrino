import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family']       = 'sans-serif'
mpl.rcParams['font.sans-serif']   = ['DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']
mpl.rcParams['mathtext.fontset']  = 'dejavusans'
mpl.rcParams['axes.unicode_minus']= False
mpl.rcParams['font.size']         = 13
mpl.rcParams['axes.labelsize']    = 14
mpl.rcParams['axes.titlesize']    = 13
mpl.rcParams['xtick.labelsize']   = 12
mpl.rcParams['ytick.labelsize']   = 12
mpl.rcParams['legend.fontsize']   = 11
mpl.rcParams['figure.dpi']        = 100

class MiniBooNE:
    def __init__(self):
#####################################加速器源
        self.source_Elo = 0

        self.source_Ehi = 0

        self.source_numu = 0

        self.source_numub = 0

        self.source_nue = 0

        self.source_nueb = 0
######################################探测器事件
        self.obs_Elo = 0

        self.obs_Ehi = 0

        self.observed = 0

        self.background = 0

        self.cov = 0
######################################探测器配置
        self.N_target = 0

        self.L_km = 0

        self.pot = 0

        self.sigma = 0

        self.efficiency = 0

        self.responsematrix = self._build_placeholder_response_matrix()
######################################数据预处理
        self.source_E_centers   = (self.source_Elo + self.source_Ehi) / 2.0
        self.n_source_bins = len(self.source_E_centers)
        self.obs_E_centers   = (self.obs_Elo + self.obs_Ehi) / 2.0
        self.n_obs_bins = len(self.obs_E_centers)

        self.flux_arrays  = {'numu': self.source_numu, 'numub': self.source_numub,
                             'nue':  self.source_nue,  'nueb':  self.source_nueb}

        self.E_true = self.source_E_centers
        self.E_true_edges = np.concatenate([[self.Elo[0]], self.Ehi])
        self.dE_true = np.diff(self.E_true_edges)

        self.flux_arrays  = {'numu': self.source_numu, 'numub': self.source_numub,
                             'nue':  self.source_nue,  'nueb':  self.source_nueb}
        self.absolute_flux = {k: v.copy() for k, v in self.flux_arrays.items()}

        #self.obs_Elo_GeV     = self.obs_Elo / 1000.
        #self.obs_Ehi_GeV     = self.obs_Ehi / 1000.
        #self.obs_centers_GeV = (self.obs_Elo_GeV + self.obs_Ehi_GeV) / 2.
        self.inv_cov  = inv(self.cov)

        self._normalize_response_matrix()


    def _build_placeholder_response_matrix(self):
        R = np.zeros((self.n_flux_bins, self.n_obs_bins))
        for i, E in enumerate(self.source_E_centers):
            for j in range(self.n_obs_bins):
                if self.obs_Elo[j] <= E < self.obs_Ehi[j]:
                    R[i, j] = 1.0;break
        return R

    def _normalize_response_matrix(self):
        rs = self.responsematrix.sum(axis=1, keepdims=True)
        nz = rs[:, 0] > 0
        self.responsematrix[nz] /= rs[nz]

    def get_energy_spectrum(self, flavor):
        return self.E_centers, self.norm_weights[flavor]
    def get_absolute_flux(self, flavor):
        return self.source_E_centers, self.absolute_flux[flavor]
    def get_target(self):
        return self.N_target
    def get_sigma_array(self):
        return self.sigma
    def get_efficiency_array(self):
        return self.efficiency
    def get_pot(self):
        return self.pot
    def get_baseline_km(self):
        return self.L_km
    def get_observations(self):
        return self.observed
    def get_background(self):
        return self.background
    def get_covariance(self):
        return self.cov
    def get_inv_covariance(self):
        return self.inv_cov
    def get_experiment_bins(self):
        return self.obs_Elo, self.obs_Ehi, self.obs_E_centers
    def get_flux_bins(self):
        return self.source_Elo, self.source_Ehi, self.source_E_centers
    def get_response_matrix(self):
        return self.responsematrix
    def get_true_bin_widths(self):
        return self.dE_true.copy()


def rotation_matrix(N, i, j, theta, delta=0.0):
    R = np.identity(N, dtype=complex)
    s, c = np.sin(theta), np.cos(theta)
    R[i,i] =  c;  R[j,j] =  c
    R[i,j] =  s * np.exp(-1j * delta)
    R[j,i] = -s * np.exp( 1j * delta)
    return R

class NeutrinoModels:
    def __init__(self):
        self.theta12    = 33  * np.pi / 180
        self.theta13    = 8.6 * np.pi / 180
        self.theta23    = 49  * np.pi / 180
        self.delta13    = 0.0
        self.m2_3flavor = np.array([0., 7.5e-5, 2.5e-3])
        self.sterile_param_names = {
            "3":     [],
            "3+1":   ["theta14","theta24","theta34","delta14","delta24","delta34",
                      "dm41"],
            "3+2":   ["theta14","theta24","theta34",
                      "theta15","theta25","theta35","theta45",
                      "delta14","delta24","delta34",
                      "delta15","delta25","delta35","delta45",
                      "dm41","dm51"],
            "1+3+1": ["theta14","theta24","theta34",
                      "theta15","theta25","theta35","theta45",
                      "delta14","delta24","delta34",
                      "delta15","delta25","delta35","delta45",
                      "dm41","dm51"],
            "3+3":   ["theta14","theta24","theta34",
                      "theta15","theta25","theta35","theta45",
                      "theta16","theta26","theta36","theta46","theta56",
                      "delta14","delta24","delta34",
                      "delta15","delta25","delta35","delta45",
                      "delta16","delta26","delta36","delta46","delta56",
                      "dm41","dm51","dm61"],
        }

    def build_model(self, model_name, params):
        b = {"3": self._build_3flavor, "3+1": self._build_3plus1,
             "3+2": self._build_3plus2, "1+3+1": self._build_1plus3plus1}
        if model_name not in b: raise ValueError(f"未知模型: {model_name}")
        return b[model_name](params) if model_name != "3" else b[model_name]()
    def _sa(self, v): return np.arcsin(np.clip(v, 0., 1.))

    def _build_3flavor(self):
        N, U = 3, (rotation_matrix(3,1,2,self.theta23) @
                   rotation_matrix(3,0,2,self.theta13,delta=self.delta13) @
                   rotation_matrix(3,0,1,self.theta12))
        return U, self.m2_3flavor.copy(), N

    def _build_3plus1(self, p):
        N = 4
        Ua = (rotation_matrix(N,1,2,self.theta23) @
              rotation_matrix(N,0,2,self.theta13,delta=self.delta13) @
              rotation_matrix(N,0,1,self.theta12))
        Us = (rotation_matrix(N,2,3,self._sa(p.get("theta34",0.)),delta=p.get("delta34",0.)) @
              rotation_matrix(N,1,3,self._sa(p.get("theta24",0.)),delta=p.get("delta24",0.)) @
              rotation_matrix(N,0,3,self._sa(p.get("theta14",0.)),delta=p.get("delta14",0.)))
        return Us@Ua, np.concatenate([self.m2_3flavor,[p.get("dm41",1.0)]]), N

    def _build_3plus2(self, p):
        N = 5
        Ua  = (rotation_matrix(N,1,2,self.theta23) @
               rotation_matrix(N,0,2,self.theta13,delta=self.delta13) @
               rotation_matrix(N,0,1,self.theta12))
        Us1 = (rotation_matrix(N,2,3,self._sa(p.get("theta34",0.)),delta=p.get("delta34",0.)) @
               rotation_matrix(N,1,3,self._sa(p.get("theta24",0.)),delta=p.get("delta24",0.)) @
               rotation_matrix(N,0,3,self._sa(p.get("theta14",0.)),delta=p.get("delta14",0.)))
        Us2 = (rotation_matrix(N,3,4,self._sa(p.get("theta45",0.)),delta=p.get("delta45",0.)) @
               rotation_matrix(N,2,4,self._sa(p.get("theta35",0.)),delta=p.get("delta35",0.)) @
               rotation_matrix(N,1,4,self._sa(p.get("theta25",0.)),delta=p.get("delta25",0.)) @
               rotation_matrix(N,0,4,self._sa(p.get("theta15",0.)),delta=p.get("delta15",0.)))
        return Us2@Us1@Ua, np.concatenate([self.m2_3flavor,[p.get("dm41",0.47),p.get("dm51",0.87)]]), N

    def _build_1plus3plus1(self, p):
        N = 5
        Ua  = (rotation_matrix(N,1,2,self.theta23) @
               rotation_matrix(N,0,2,self.theta13,delta=self.delta13) @
               rotation_matrix(N,0,1,self.theta12))
        Us1 = (rotation_matrix(N,2,3,self._sa(p.get("theta34",0.)),delta=p.get("delta34",0.)) @
               rotation_matrix(N,1,3,self._sa(p.get("theta24",0.)),delta=p.get("delta24",0.)) @
               rotation_matrix(N,0,3,self._sa(p.get("theta14",0.)),delta=p.get("delta14",0.)))
        Us2 = (rotation_matrix(N,4,5,self._sa(p.get("theta56",0.)),delta=p.get("delta56",0.)) @
               rotation_matrix(N,3,5,self._sa(p.get("theta46",0.)),delta=p.get("delta46",0.)) @
               rotation_matrix(N,2,5,self._sa(p.get("theta36",0.)),delta=p.get("delta36",0.)) @
               rotation_matrix(N,1,5,self._sa(p.get("theta26",0.)),delta=p.get("delta26",0.)) @
               rotation_matrix(N,0,5,self._sa(p.get("theta16",0.)),delta=p.get("delta16",0.)))
        return Us2@Us1@Ua, np.concatenate([self.m2_3flavor,[p.get("dm41",-0.87),p.get("dm51",0.47)]]), N

    def _build_1plus3plus3(self, p):
        N = 6
        Ua  = (rotation_matrix(N,1,2,self.theta23) @
               rotation_matrix(N,0,2,self.theta13,delta=self.delta13) @
               rotation_matrix(N,0,1,self.theta12))
        Us1 = (rotation_matrix(N,2,3,self._sa(p.get("theta34",0.)),delta=p.get("delta34",0.)) @
               rotation_matrix(N,1,3,self._sa(p.get("theta24",0.)),delta=p.get("delta24",0.)) @
               rotation_matrix(N,0,3,self._sa(p.get("theta14",0.)),delta=p.get("delta14",0.)))
        Us2 = (rotation_matrix(N,3,4, self._sa(p.get("theta45", 0.)), delta=p.get("delta45", 0.)) @
               rotation_matrix(N,2,4,self._sa(p.get("theta35",0.)),delta=p.get("delta35",0.)) @
               rotation_matrix(N,1,4,self._sa(p.get("theta25",0.)),delta=p.get("delta25",0.)) @
               rotation_matrix(N,0,4,self._sa(p.get("theta15",0.)),delta=p.get("delta15",0.)))
        Us3 = (rotation_matrix(N,3,4, self._sa(p.get("theta45", 0.)), delta=p.get("delta45", 0.)) @
               rotation_matrix(N,2,4, self._sa(p.get("theta35", 0.)), delta=p.get("delta35", 0.)) @
               rotation_matrix(N,1,4, self._sa(p.get("theta25", 0.)), delta=p.get("delta25", 0.)) @
               rotation_matrix(N,0,4, self._sa(p.get("theta15", 0.)), delta=p.get("delta15", 0.)))
        return Us3@Us2@Us1 @ Ua, np.concatenate([self.m2_3flavor, [p.get("dm41", 1.0), p.get("dm51", 1.0), p.get("dm61", 1.0)]]), N

class OscillationModelWrapper:
    def __init__(self):
        self.model_builder = NeutrinoModels()
    def get_mixing_and_masses(self, model_name, params):
        return self.model_builder.build_model(model_name, params)

def oscillation_probability_array(U, m2, alpha, beta, L, E_arr, anti=False):
    phase  = np.exp(-1j * 1.267 * m2[np.newaxis,:] * L / E_arr[:,np.newaxis])
    coeffs = (np.conj(U[beta,:]) * U[alpha,:] if anti else U[beta,:] * np.conj(U[alpha,:]))
    return np.abs(phase @ coeffs) ** 2

class EventRateCalculator:
    def __init__(self, exp, model_wrapper):
        self.exp     = exp
        self.model   = model_wrapper
        self.L_km    = exp.L_km
        self.E_true  = exp.get_baseline_km()
        self.R       = exp.get_response_matrix()
        self.pc_nu   = exp.precomp_nu
        self.pc_nub  = exp.precomp_nub

    def compute_true_spectrum(self, U, m2):
        P_nu  = oscillation_probability_array(U, m2, 1, 0, self.L_km, self.E_true, False)
        P_nub = oscillation_probability_array(U, m2, 1, 0, self.L_km, self.E_true, True)
        return self.pc_nu * P_nu + self.pc_nub * P_nub

    def compute_signal_spectrum(self, model_name, params):
        U, m2, _ = self.model.get_mixing_and_masses(model_name, params)
        return self.R.T @ self.compute_true_spectrum(U, m2)

# ============================================================
# 似然计算器
# ============================================================
class LikelihoodCalculator:

    def __init__(self, exp, model_wrapper, model_name, param_settings):
        self.exp           = exp
        self.model_wrapper = model_wrapper
        self.model_name    = model_name
        self.param_names   = model_wrapper.model_builder.sterile_param_names[model_name]
        self.param_info    = {}
        self.param_scale   = {}

        for nm in self.param_names:
            if nm not in param_settings:
                raise KeyError(f"参数 '{nm}' 未在 param_settings 中定义。")
            cfg = dict(param_settings[nm])
            self.param_scale[nm] = cfg.get('scale', 'linear')
            self.param_info[nm]  = cfg

        self.rate_calc           = EventRateCalculator(exp, model_wrapper)
        self._chi2_global_min    = None
        self._global_best_params = None

    def _to_opt(self, v, nm):
        return np.log10(max(float(v), 1e-12)) if self.param_scale[nm] == 'log' else float(v)

    def _from_opt(self, v, nm):
        return 10.**float(v) if self.param_scale[nm] == 'log' else float(v)

    def _v2p(self, x):
        return {nm: self._from_opt(x[i], nm) for i, nm in enumerate(self.param_names)}

    def get_bounds(self):
        out = []
        for nm in self.param_names:
            lo, hi = self.param_info[nm]['lower'], self.param_info[nm]['upper']
            if self.param_scale[nm] == 'log':
                lo = np.log10(max(lo, 1e-12))
                hi = np.log10(max(hi, 1e-12))
            out.append((lo, hi))
        return out

    def get_initial_guess(self):
        return np.array([self._to_opt(self.param_info[nm]['init'], nm)
                         for nm in self.param_names])

    def get_init_param_dict(self):
        return {nm: float(self.param_info[nm]['init']) for nm in self.param_names}

    def log_likelihood(self, x):
        p     = self._v2p(x)
        n_obs = self.exp.get_observations()
        bkg   = self.exp.get_background()
        cov_f = self.exp.get_covariance()
        S     = self.rate_calc.compute_signal_spectrum(self.model_name, p)
        mu    = S + bkg
        mu    = np.maximum(mu, 1e-12)
        cov_a = cov_f * np.outer(mu, mu)
        n     = cov_a.shape[0]
        eps   = 1e-8 * np.trace(cov_a) / n
        cov_r = cov_a + eps * np.eye(n)
        try:    inv_c = np.linalg.inv(cov_r)
        except: inv_c = np.linalg.pinv(cov_r)
        d = n_obs - mu
        return 0.5 * float(d @ inv_c @ d)

    def compute_chi2_null(self):
        n_obs = self.exp.get_observations()
        bkg   = self.exp.get_background()
        cov_f = self.exp.get_covariance()
        cov_a = cov_f * np.outer(bkg, bkg)
        n     = cov_a.shape[0]
        eps   = 1e-8 * np.trace(cov_a) / n
        cov_r = cov_a + eps * np.eye(n)
        try:    inv_c = np.linalg.inv(cov_r)
        except: inv_c = np.linalg.pinv(cov_r)
        d = n_obs - bkg
        return float(d @ inv_c @ d)

    def global_minimize(self, n_random=50, seed=42):
        rng    = np.random.default_rng(seed)
        bounds = self.get_bounds()
        x0     = self.get_initial_guess()

        starts = [x0]
        for _ in range(15):
            noise = rng.normal(0., 0.4, size=x0.shape)
            starts.append(np.clip(x0 + noise,
                                  [lo for lo, hi in bounds],
                                  [hi for lo, hi in bounds]))
        for _ in range(5):
            pt = x0.copy()
            for k, nm in enumerate(self.param_names):
                if nm.startswith('theta') and self.param_info[nm]['upper'] > 0.01:
                    lo_ph, hi_ph = 0.05, 0.5
                    if self.param_scale[nm] == 'log':
                        pt[k] = rng.uniform(np.log10(lo_ph), np.log10(hi_ph))
                    else:
                        pt[k] = rng.uniform(lo_ph, hi_ph)
            pt = np.clip(pt, [lo for lo, hi in bounds], [hi for lo, hi in bounds])
            starts.append(pt)
        for _ in range(n_random):
            starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

        best_x, best_c2 = None, np.inf
        opt = dict(method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-9})
        for xs in starts:
            try:
                res = minimize(self.log_likelihood, xs, **opt)
                c2  = 2. * res.fun
                if c2 < best_c2:
                    best_c2 = c2; best_x = res.x.copy()
            except Exception:
                continue

        if best_x is None:
            best_x = x0; best_c2 = 2. * self.log_likelihood(x0)

        try:
            res2 = minimize(self.log_likelihood, best_x, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if 2. * res2.fun < best_c2:
                best_c2 = 2. * res2.fun; best_x = res2.x.copy()
        except Exception:
            pass

        best_params = self._v2p(best_x)
        self._chi2_global_min    = float(best_c2)
        self._global_best_params = best_params
        return best_params, float(best_c2)

    def profile_likelihood(self, fixed_params, x0_free=None, n_random_starts=3, seed=None):
        fx_idx    = [i for i, nm in enumerate(self.param_names) if nm in fixed_params]
        fx_vals   = [fixed_params[nm] for nm in self.param_names if nm in fixed_params]
        fr_idx    = [i for i in range(len(self.param_names)) if i not in fx_idx]
        fr_nms    = [self.param_names[i] for i in fr_idx]
        bounds    = self.get_bounds()
        fr_bounds = [bounds[i] for i in fr_idx]

        def obj(free_x):
            full = np.zeros(len(self.param_names))
            for idx, val in zip(fx_idx, fx_vals):
                full[idx] = self._to_opt(val, self.param_names[idx])
            full[fr_idx] = free_x
            return self.log_likelihood(full)

        starts = []
        starts.append(np.array([self._to_opt(self.param_info[nm]['init'], nm) for nm in fr_nms]))
        if x0_free is not None:
            starts.append(np.array(x0_free))
        if self._global_best_params is not None:
            gx = np.array([self._to_opt(self._global_best_params.get(nm, self.param_info[nm]['init']), nm)
                           for nm in fr_nms])
            starts.append(np.clip(gx, [lo for lo, hi in fr_bounds], [hi for lo, hi in fr_bounds]))
        rng = np.random.default_rng(seed)
        for _ in range(n_random_starts):
            starts.append(np.array([rng.uniform(lo, hi) for lo, hi in fr_bounds]))

        opt = dict(method='L-BFGS-B', bounds=fr_bounds,
                   options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-9})
        best_fun, best_x = np.inf, starts[0]
        for xs in starts:
            try:
                res = minimize(obj, xs, **opt)
                if res.fun < best_fun:
                    best_fun = res.fun; best_x = res.x.copy()
            except Exception:
                continue

        full_opt = np.zeros(len(self.param_names))
        for idx, val in zip(fx_idx, fx_vals):
            full_opt[idx] = self._to_opt(val, self.param_names[idx])
        full_opt[fr_idx] = best_x
        return self._v2p(full_opt), -best_fun

































