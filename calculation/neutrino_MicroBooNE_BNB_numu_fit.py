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

# ============================================================
# 实验数据基类
# ============================================================
class ExperimentBase:
    def get_energy_spectrum(self, flavor):  return self.E_centers, self.norm_weights[flavor]
    def get_absolute_flux(self, flavor):    return self.E_centers, self.absolute_flux[flavor]
    def get_target(self):                   return self.N_target
    def get_sigma_array(self):              return self.sigma_flux
    def get_efficiency_array(self):         return self.efficiency_flux
    def get_pot(self):                      return self.pot
    def get_baseline_km(self):              return self.L_km
    def get_observations(self):             return self.observed.copy()
    def get_background(self):              return self.background.copy()
    def get_covariance(self):               return self.covariance.copy()
    def get_inv_covariance(self):           return self.inv_covariance.copy()
    def get_experiment_bins(self):          return (self.obs_Elo_GeV, self.obs_Ehi_GeV, self.obs_centers_GeV)
    def get_flux_bins(self):                return (self.Elo, self.Ehi, self.E_centers)
    def get_response_matrix(self):          return self.response_matrix.copy()
    def get_true_bin_widths(self):          return self.dE_true.copy()


class MicroBooNE(ExperimentBase):
    def __init__(self):
        self.Elo = np.arange(0.00, 10.00, 0.05)
        self.Ehi = self.Elo + 0.05

        self.numu_flux = np.array([
            2.272e-12, 8.566e-12, 1.112e-11, 1.335e-11, 1.658e-11, 1.820e-11, 1.946e-11, 2.045e-11, 2.161e-11, 2.241e-11,
            2.279e-11, 2.292e-11, 2.275e-11, 2.253e-11, 2.214e-11, 2.156e-11, 2.078e-11, 1.992e-11, 1.894e-11, 1.789e-11,
            1.677e-11, 1.558e-11, 1.439e-11, 1.318e-11, 1.193e-11, 1.069e-11, 9.503e-12, 8.356e-12, 7.278e-12, 6.292e-12,
            5.396e-12, 4.601e-12, 3.902e-12, 3.285e-12, 2.760e-12, 2.312e-12, 1.932e-12, 1.616e-12, 1.355e-12, 1.138e-12,
            9.589e-13, 8.150e-13, 6.928e-13, 5.937e-13, 5.147e-13, 4.478e-13, 3.935e-13, 3.500e-13, 3.150e-13, 2.867e-13,
            2.615e-13, 2.409e-13, 2.273e-13, 2.110e-13, 1.995e-13, 1.920e-13, 1.815e-13, 1.726e-13, 1.665e-13, 1.601e-13,
            1.554e-13, 1.493e-13, 1.442e-13, 1.412e-13, 1.363e-13, 1.323e-13, 1.265e-13, 1.217e-13, 1.183e-13, 1.140e-13,
            1.102e-13, 1.060e-13, 1.014e-13, 9.700e-14, 9.340e-14, 9.001e-14, 8.641e-14, 8.190e-14, 7.867e-14, 7.464e-14,
            7.146e-14, 6.812e-14, 6.499e-14, 6.185e-14, 5.858e-14, 5.614e-14, 5.320e-14, 5.016e-14, 4.765e-14, 4.561e-14,
            4.281e-14, 4.087e-14, 3.841e-14, 3.632e-14, 3.432e-14, 3.263e-14, 3.016e-14, 2.857e-14, 2.689e-14, 2.529e-14,
            2.372e-14, 2.227e-14, 2.103e-14, 1.957e-14, 1.834e-14, 1.730e-14, 1.615e-14, 1.513e-14, 1.406e-14, 1.303e-14,
            1.214e-14, 1.129e-14, 1.047e-14, 9.569e-15, 8.870e-15, 8.148e-15, 7.429e-15, 6.765e-15, 6.097e-15, 5.492e-15,
            4.977e-15, 4.445e-15, 3.967e-15, 3.492e-15, 3.037e-15, 2.595e-15, 2.225e-15, 1.854e-15, 1.537e-15, 1.220e-15,
            9.780e-16, 7.842e-16, 6.198e-16, 4.786e-16, 3.334e-16, 1.971e-16, 9.391e-17, 2.738e-17, 6.065e-18, 4.135e-18,
            1.933e-18, 9.888e-19, 4.494e-20, *([0.0]*57)])

        self.numub_flux = np.array([
            2.560e-12, 5.671e-12, 3.300e-12, 2.028e-12, 1.623e-12, 1.395e-12, 1.301e-12, 1.249e-12, 1.171e-12, 1.054e-12,
            9.580e-13, 8.695e-13, 8.098e-13, 7.434e-13, 6.910e-13, 6.314e-13, 5.905e-13, 5.504e-13, 5.079e-13, 4.708e-13,
            4.347e-13, 4.021e-13, 3.703e-13, 3.443e-13, 3.173e-13, 2.872e-13, 2.597e-13, 2.337e-13, 2.101e-13, 1.903e-13,
            1.718e-13, 1.507e-13, 1.341e-13, 1.173e-13, 1.053e-13, 9.241e-14, 8.188e-14, 7.115e-14, 6.349e-14, 5.547e-14,
            4.799e-14, 4.071e-14, 3.592e-14, 3.082e-14, 2.638e-14, 2.248e-14, 1.878e-14, 1.623e-14, 1.391e-14, 1.162e-14,
            1.010e-14, 8.691e-15, 7.382e-15, 5.999e-15, 5.004e-15, 4.204e-15, 3.571e-15, 3.047e-15, 2.597e-15, 2.138e-15,
            1.956e-15, 1.584e-15, 1.227e-15, 1.021e-15, 8.356e-16, 7.777e-16, 6.812e-16, 7.386e-16, 6.128e-16, 6.251e-16,
            5.519e-16, 3.936e-16, 4.141e-16, 3.395e-16, 3.002e-16, 2.502e-16, 2.273e-16, 2.299e-16, 1.429e-16, 1.574e-16,
            1.218e-16, 1.280e-16, 1.612e-16, 8.604e-17, 9.270e-17, 5.371e-17, 5.495e-17, 4.276e-17, 3.693e-17, 6.592e-17,
            6.261e-17, 2.266e-17, 3.924e-17, 5.036e-17, 3.051e-17, 7.985e-17, 1.630e-16, 1.787e-16, 5.729e-17, 6.383e-18,
            5.257e-18, 5.222e-18, 4.369e-18, 3.186e-18, 3.915e-18, 2.197e-18, 1.690e-18, 1.177e-18, 9.963e-19, 9.197e-19,
            6.790e-19, 5.695e-19, 5.234e-19, 3.209e-19, 2.809e-19, 2.700e-19, 1.624e-19, 1.383e-19, 1.192e-19, 9.024e-20,
            9.442e-20, 5.076e-20, 6.390e-20, 4.695e-20, 2.734e-20, 3.940e-20, 2.067e-20, 2.327e-20, 2.294e-20, 1.385e-20,
            1.932e-21, 8.299e-21, 5.854e-21, 1.843e-21, 0., 1.783e-21, 4.490e-21, 4.205e-21, *([0.0]*62)])

        self.nue_flux = np.array([
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
            1.571e-19, 1.169e-19, *([0.0]*58)])

        self.nueb_flux = np.array([
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
            1.993e-19, 1.595e-19, 3.987e-20, *([0.0]*57)])

        self._post_init_flux()

        self.obs_Elo = np.linspace(0.0, 2.5, 26)
        self.obs_Ehi = np.r_[np.linspace(0.1, 2.5, 25), 10]

        self.observed   = np.array([83, 90, 63, 58, 49, 45, 35, 67], dtype=float)
        self.background = np.array([70.91, 81.51, 61.94, 58.95, 50.01, 44.49, 33.73, 65.27], dtype=float)

        cov_raw = [
            [ 0.022008,   0.0012446,  0.0048977,  0.0034062,  0.0044717, -0.0028033,  0.00040001, 0.0020426],
            [ 0.0012446,  0.02618,    0.0012558,  0.0050882,  0.00094575, 0.0076589,  0.0036528,  0.011599 ],
            [ 0.0048977,  0.0012558,  0.026245,   0.013242,   0.015989,   0.013736,   0.018327,   0.016744 ],
            [ 0.0034062,  0.0050882,  0.013242,   0.034777,   0.023469,   0.024032,   0.033912,   0.026725 ],
            [ 0.0044717,  0.00094575, 0.015989,   0.023469,   0.052856,   0.032156,   0.05459,    0.028499 ],
            [-0.0028033,  0.0076589,  0.013736,   0.024032,   0.032156,   0.082711,   0.069783,   0.050257 ],
            [ 0.00040001, 0.0036528,  0.018327,   0.033912,   0.05459,    0.069783,   0.12506,    0.053062 ],
            [ 0.0020426,  0.011599,   0.016744,   0.026725,   0.028499,   0.050257,   0.053062,   0.064412 ]
        ]
        self.covariance = np.array(cov_raw, dtype=float)
        self._post_init_obs()

        self.N_target        = 1.3e30
        self.L_km            = 0.541
        self.pot             = 6.46e20
        self.sigma_flux      = np.full(self.n_flux_bins, 1e-38)
        self.efficiency_flux = np.full(self.n_flux_bins, 0.20)

        self.E_true       = self.E_centers
        self.E_true_edges = np.concatenate([[self.Elo[0]], self.Ehi])
        self.dE_true      = np.diff(self.E_true_edges)

        self.response_matrix = self._build_placeholder_response_matrix()
        self._normalize_response_matrix()

        self.precomp_nu  = self.absolute_flux['numu']  * self.sigma_flux * self.efficiency_flux * self.pot * self.N_target
        self.precomp_nub = self.absolute_flux['numub'] * self.sigma_flux * self.efficiency_flux * self.pot * self.N_target

    def _post_init_obs(self):
        self.obs_Elo_GeV     = self.obs_Elo / 1000.
        self.obs_Ehi_GeV     = self.obs_Ehi / 1000.
        self.obs_centers_GeV = (self.obs_Elo_GeV + self.obs_Ehi_GeV) / 2.
        self.n_obs_bins      = len(self.obs_centers_GeV)
        self.inv_covariance  = inv(self.covariance)

    def _post_init_flux(self):
        self.E_centers   = (self.Elo + self.Ehi) / 2.0
        self.n_flux_bins = len(self.E_centers)
        self.flux_arrays  = {'numu': self.numu_flux, 'numub': self.numub_flux,
                             'nue':  self.nue_flux,  'nueb':  self.nueb_flux}
        self.absolute_flux = {k: v.copy() for k, v in self.flux_arrays.items()}
        self.norm_weights  = {}
        for fl, arr in self.flux_arrays.items():
            s = arr.sum()
            self.norm_weights[fl] = arr / s if s > 0 else np.zeros_like(arr)

    def _post_init_obs(self):
        self.obs_Elo_GeV     = self.obs_Elo / 1000.
        self.obs_Ehi_GeV     = self.obs_Ehi / 1000.
        self.obs_centers_GeV = (self.obs_Elo_GeV + self.obs_Ehi_GeV) / 2.
        self.n_obs_bins      = len(self.obs_centers_GeV)
        self.inv_covariance  = inv(self.covariance)

    def _build_placeholder_response_matrix(self):
        R = np.zeros((self.n_flux_bins, self.n_obs_bins))
        for i, E in enumerate(self.E_centers):
            for j in range(self.n_obs_bins):
                if self.obs_Elo_GeV[j] <= E < self.obs_Ehi_GeV[j]:
                    R[i, j] = 1.0; break
        return R

    def _normalize_response_matrix(self):
        rs = self.response_matrix.sum(axis=1, keepdims=True)
        nz = rs[:, 0] > 0
        self.response_matrix[nz] /= rs[nz]


# ============================================================
# 旋转矩阵 / 振荡模型
# ============================================================
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
            "3": [],
            "3+1": ["theta14","theta24","theta34","delta14","delta24","delta34","dm41"],
            "3+2": ["theta14","theta24","theta34","theta15","theta25","theta35","theta45",
                    "delta14","delta24","delta34","delta15","delta25","delta35","delta45","dm41","dm51"],
            "1+3+1": ["theta14","theta24","theta34","theta15","theta25","theta35","theta45",
                      "delta14","delta24","delta34","delta15","delta25","delta35","delta45","dm41","dm51"]
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
        Us2 = (rotation_matrix(N,3,4,self._sa(p.get("theta45",0.)),delta=p.get("delta45",0.)) @
               rotation_matrix(N,2,4,self._sa(p.get("theta35",0.)),delta=p.get("delta35",0.)) @
               rotation_matrix(N,1,4,self._sa(p.get("theta25",0.)),delta=p.get("delta25",0.)) @
               rotation_matrix(N,0,4,self._sa(p.get("theta15",0.)),delta=p.get("delta15",0.)))
        return Us2@Us1@Ua, np.concatenate([self.m2_3flavor,[p.get("dm41",-0.87),p.get("dm51",0.47)]]), N


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
        self.L_km    = exp.get_baseline_km()
        self.E_true  = exp.E_true
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

    @staticmethod
    def params_to_plot_coords(params, mode):
        t14 = params.get('theta14', 0.)
        t24 = params.get('theta24', 0.)
        t15 = params.get('theta15', 0.)
        t25 = params.get('theta25', 0.)
        d41 = abs(params.get('dm41', 0.))
        d51 = abs(params.get('dm51', 0.))
        if   mode == 1: return t14 * t24, t15 * t25
        elif mode == 2: return d41, d51
        elif mode == 3: return 4. * t14**2 * t24**2, d41
        else: raise ValueError(f"不支持 mode={mode}")

    def scan_physical_2d(self, grid1, grid2, mode=1,
                         fixed_base=None,
                         chi2_null_override=None):
        if fixed_base is None:
            fixed_base = {}

        n1, n2      = len(grid1), len(grid2)
        chi2_grid   = np.full((n1, n2), np.nan)
        param_store = [[None] * n2 for _ in range(n1)]
        tmpl        = {nm: float(self.param_info[nm]['init']) for nm in self.param_names}

        for i, v1 in enumerate(grid1):
            for j, v2 in enumerate(grid2):
                p = dict(tmpl)
                _mode_keys = self._get_mode_scan_keys(mode)
                for k, val in fixed_base.items():
                    if k in p and k not in _mode_keys:
                        p[k] = val
                self._apply_mode_scan_values(p, mode, v1, v2, fixed_base)
                x = np.array([self._to_opt(p[nm], nm) for nm in self.param_names])
                chi2_grid[i, j]    = 2. * self.log_likelihood(x)
                param_store[i][j]  = dict(p)

        chi2_scan_min = float(np.nanmin(chi2_grid))
        TS_grid       = np.maximum(0., chi2_grid - chi2_scan_min)
        assert float(np.nanmin(TS_grid)) < 1e-6, "TS_min > 0，TS 基准错误。"

        idx_best         = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
        best_full_params = param_store[idx_best[0]][idx_best[1]]
        best_xy          = (float(grid1[idx_best[0]]), float(grid2[idx_best[1]]))
        return chi2_grid, TS_grid, best_xy, chi2_scan_min, best_full_params

    def scan_physical_2d_profile(self, grid1, grid2, mode=1,
                                  fixed_base=None,
                                  chi2_null_override=None):
        assert self._chi2_global_min is not None, \
            "必须先调用 global_minimize() 才能运行 scan_physical_2d_profile()。"

        if fixed_base is None:
            fixed_base = {}

        n1, n2      = len(grid1), len(grid2)
        chi2_grid   = np.full((n1, n2), np.nan)
        param_store = [[None] * n2 for _ in range(n1)]

        for i, v1 in enumerate(grid1):
            for j, v2 in enumerate(grid2):
                fixed_params       = self._build_fixed_params_for_profile(mode, v1, v2, fixed_base)
                _, neg_logL        = self.profile_likelihood(fixed_params)
                chi2_grid[i, j]    = -2. * neg_logL
                param_store[i][j]  = dict(fixed_params)

        TS_grid          = np.maximum(0., chi2_grid - self._chi2_global_min)
        idx_best         = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
        best_full_params = param_store[idx_best[0]][idx_best[1]]
        best_xy          = (float(grid1[idx_best[0]]), float(grid2[idx_best[1]]))
        return chi2_grid, TS_grid, best_xy, self._chi2_global_min, best_full_params

    def _build_fixed_params_for_profile(self, mode, v1, v2, fixed_base):
        fp = {}
        if mode == 1:
            sin14 = float(np.clip(np.sqrt(max(v1, 0.)), 0., 1.))
            sin15 = float(np.clip(np.sqrt(max(v2, 0.)), 0., 1.))
            fp['theta14'] = sin14; fp['theta24'] = sin14
            fp['theta15'] = sin15; fp['theta25'] = sin15
        elif mode == 2:
            fp['dm41'] = float(v1)
            fp['dm51'] = float(v2)
        elif mode == 3:
            s2t = float(np.clip(v1, 1e-9, 1.))
            st  = float(np.clip((s2t / 4.) ** 0.25, 0., 1.))
            fp['theta14'] = st; fp['theta24'] = st
            fp['dm41']    = float(v2)
        else:
            raise ValueError(f"不支持 mode={mode}（有效模式：1, 2, 3）")
        return fp

    @staticmethod
    def _get_mode_scan_keys(mode):
        if mode == 1:   return {'theta14', 'theta24', 'theta15', 'theta25'}
        elif mode == 2: return {'dm41', 'dm51'}
        elif mode == 3: return {'theta14', 'theta24', 'dm41'}
        else: raise ValueError(f"不支持 mode={mode}（有效模式：1, 2, 3）")

    @staticmethod
    def _apply_mode_scan_values(p, mode, v1, v2, fixed_base):
        if mode == 1:
            sin14 = float(np.clip(np.sqrt(max(v1, 0.)), 0., 1.))
            sin15 = float(np.clip(np.sqrt(max(v2, 0.)), 0., 1.))
            p['theta14'] = sin14; p['theta24'] = sin14
            p['theta15'] = sin15; p['theta25'] = sin15
        elif mode == 2:
            p['dm41'] = float(v1)
            p['dm51'] = float(v2)
        elif mode == 3:
            s2t = float(np.clip(v1, 1e-9, 1.))
            st  = float(np.clip((s2t / 4.) ** 0.25, 0., 1.))
            p['theta14'] = st; p['theta24'] = st
            p['dm41']    = float(v2)
        else:
            raise ValueError(f"不支持 mode={mode}（有效模式：1, 2, 3）")


# ============================================================
# 全局 best-fit 投影点 TS 计算
# ============================================================
def _compute_ts_at_projection(g1, g2, chi2_grid, chi2_ref, proj_xy,
                               xscale='log', yscale='log'):
    px, py = proj_xy
    if px is None or py is None: return np.inf
    if not np.isfinite(px) or not np.isfinite(py): return np.inf
    if px <= 0 or py <= 0: return np.inf

    tol = 1.05
    if xscale == 'log':
        if px < g1[0] / tol or px > g1[-1] * tol: return np.inf
    else:
        rng = g1[-1] - g1[0]
        if px < g1[0] - 0.05*rng or px > g1[-1] + 0.05*rng: return np.inf
    if yscale == 'log':
        if py < g2[0] / 1.05 or py > g2[-1] * 1.05: return np.inf
    else:
        rng = g2[-1] - g2[0]
        if py < g2[0] - 0.05*rng or py > g2[-1] + 0.05*rng: return np.inf

    di = int(np.argmin(np.abs(np.log10(g1) - np.log10(px)))) if xscale == 'log' else int(np.argmin(np.abs(g1 - px)))
    dj = int(np.argmin(np.abs(np.log10(g2) - np.log10(py)))) if yscale == 'log' else int(np.argmin(np.abs(g2 - py)))
    c2 = chi2_grid[di, dj]
    return np.inf if np.isnan(c2) else float(max(0., c2 - chi2_ref))


# ============================================================
# 绘图函数
# ============================================================
def _mode_axis_info(mode):
    info = {
        1: dict(xlabel=r"$\sin\theta_{14}\cdot\sin\theta_{24}$",
                ylabel=r"$\sin\theta_{15}\cdot\sin\theta_{25}$",
                xscale='log', yscale='log',
                title=r"$\sin\theta_{14}\sin\theta_{24}$ vs $\sin\theta_{15}\sin\theta_{25}$"),
        2: dict(xlabel=r"$|\Delta m^2_{41}|\ \rm[eV^2]$",
                ylabel=r"$|\Delta m^2_{51}|\ \rm[eV^2]$",
                xscale='log', yscale='log',
                title=r"$|\Delta m^2_{41}|$ vs $|\Delta m^2_{51}|$"),
        3: dict(xlabel=r"$\sin^2(2\theta_{\mu e}^{\rm eff})$",
                ylabel=r"$|\Delta m^2_{41}|\ \rm[eV^2]$",
                xscale='log', yscale='log',
                title=r"$\sin^2(2\theta_{\mu e}^{\rm eff})$ vs $|\Delta m^2_{41}|$"),
    }
    if mode not in info: raise ValueError(f"不支持 mode={mode}（有效模式：1, 2, 3）")
    return info[mode]


def plot_2d_contour(grid1, grid2, TS_matrix, mode=1,
                    best_xy=None,
                    global_best_xy=None, ts_at_global_best=None,
                    chi2_null=None, chi2_scan_min=None, chi2_global_min=None,
                    model_label="MiniBooNE 1+3+1", savepath=None):
    from matplotlib.lines import Line2D

    ai     = _mode_axis_info(mode)
    title  = f"{model_label} — {ai['title']}"
    crit68, crit90, crit99 = [chi2_dist.ppf(cl, df=2) for cl in [0.68, 0.90, 0.99]]
    chi2_crits = [crit68, crit90, crit99]
    cl_labels  = ['68% CL', '90% CL', '99% CL']
    cl_ls      = [':', '-', '--']
    ts_max     = float(np.nanmax(TS_matrix))

    sq=5.5; cb_gap=0.10; cb_w=0.22; pad_l=0.85; pad_r=0.35; pad_t=0.75; pad_b=0.80; gap=1.5
    fw = pad_l+sq+cb_gap+cb_w+gap+sq+pad_r;  fh = pad_t+sq+pad_b
    fig  = plt.figure(figsize=(fw, fh))
    l1   = pad_l/fw;  b0=pad_b/fh;  w_ax=sq/fw;  h_ax=sq/fh
    l_cb = (pad_l+sq+cb_gap)/fw;    w_cb=cb_w/fw
    l2   = (pad_l+sq+cb_gap+cb_w+gap)/fw
    ax1  = fig.add_axes([l1,   b0, w_ax, h_ax])
    cax1 = fig.add_axes([l_cb, b0, w_cb, h_ax])
    ax2  = fig.add_axes([l2,   b0, w_ax, h_ax])

    X, Y = np.meshgrid(grid1, grid2, indexing='ij')

    vmax_h = min(ts_max * 1.1, chi2_crits[-1] * 4.) if ts_max > 0 else 10.
    pcm  = ax1.pcolormesh(X, Y, TS_matrix, shading='auto', cmap='viridis_r',
                          vmin=0., vmax=vmax_h)
    cb   = fig.colorbar(pcm, cax=cax1)
    cb.set_label(r'$\Delta\chi^2$', fontsize=14, labelpad=8)
    cb.ax.tick_params(labelsize=12)
    for level, ls in zip(chi2_crits, cl_ls):
        if ts_max >= level:
            ax1.contour(X, Y, TS_matrix, levels=[level], colors=['r'],
                        linestyles=[ls], linewidths=1.8, zorder=3)

    if best_xy is not None:
        ax1.plot(best_xy[0], best_xy[1], '*', color='gold', markersize=14,
                 markeredgecolor='k', markeredgewidth=0.8, zorder=7,
                 label=f'Best fit (TS≡0)\n({best_xy[0]:.3g}, {best_xy[1]:.3g})')
        ax1.legend(fontsize=10, loc='upper right', framealpha=0.85)

    ax1.set_xscale(ai['xscale']); ax1.set_yscale(ai['yscale'])
    ax1.set_xlabel(ai['xlabel'], fontsize=14); ax1.set_ylabel(ai['ylabel'], fontsize=14)
    ax1.set_title(title + r'  $\Delta\chi^2$ map', fontsize=12, pad=8)
    ax1.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.25)

    legend_handles = []
    for level, lbl, ls, fa in zip(chi2_crits, cl_labels, cl_ls, [0.15, 0.10, 0.05]):
        if ts_max >= level:
            try:
                ax2.contour(X, Y, TS_matrix, levels=[level], colors=['r'],
                            linestyles=[ls], linewidths=2.)
                ax2.contourf(X, Y, TS_matrix, levels=[0., level], colors=['r'], alpha=fa)
                legend_handles.append(Line2D([0],[0], color='r', linestyle=ls,
                                             linewidth=2., label=lbl))
            except Exception:
                legend_handles.append(Line2D([0],[0], color='#aaa', linestyle=ls,
                                             linewidth=1., label=f'{lbl} (绘制失败)'))
        else:
            legend_handles.append(Line2D([0],[0], color='#aaa', linestyle=ls,
                                         linewidth=1., label=f'{lbl} (不可见)'))

    if best_xy is not None:
        pt, = ax2.plot(best_xy[0], best_xy[1], '*', color='gold', markersize=14,
                       markeredgecolor='k', markeredgewidth=0.8, zorder=7,
                       label='Best fit (TS≡0)')
        legend_handles.append(pt)

    show_global = (global_best_xy is not None and
                   ts_at_global_best is not None and
                   np.isfinite(ts_at_global_best) and
                   ts_at_global_best < crit68)
    if show_global:
        gpt, = ax2.plot(global_best_xy[0], global_best_xy[1], 'D',
                        color='cyan', markersize=9,
                        markeredgecolor='k', markeredgewidth=0.8, zorder=6,
                        label=f'Global best proj. (TS={ts_at_global_best:.2f})')
        legend_handles.append(gpt)

    lines = []
    if chi2_null       is not None: lines.append(r'$\chi^2_{\rm null}='      + f'{chi2_null:.2f}$')
    if chi2_scan_min   is not None: lines.append(r'$\chi^2_{\rm scan\,min}=' + f'{chi2_scan_min:.2f}$')
    if chi2_global_min is not None: lines.append(r'$\chi^2_{\rm global}='    + f'{chi2_global_min:.2f}$')
    if chi2_null is not None and chi2_scan_min is not None:
        lines.append(r'$\Delta\chi^2_{\rm sig}=' + f'{chi2_null-chi2_scan_min:.2f}$')
    if ts_at_global_best is not None and not show_global and np.isfinite(ts_at_global_best):
        lines.append(r'Global proj. $\Delta{\rm TS}=' + f'{ts_at_global_best:.2f}$')
    if lines:
        ax2.text(0.03, 0.03, '\n'.join(lines), transform=ax2.transAxes,
                 fontsize=10, ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='gray', alpha=0.82))

    ax2.set_xscale(ai['xscale']); ax2.set_yscale(ai['yscale'])
    ax2.set_xlabel(ai['xlabel'], fontsize=14); ax2.set_ylabel(ai['ylabel'], fontsize=14)
    ax2.set_title(title + '  Allowed regions', fontsize=12, pad=8)
    ax2.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=0.85)
    ax2.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
    ax2.grid(True, which='both', linestyle='--', alpha=0.25)

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================
PLOT_MODE = 1
ALL_MODES = True


def main():
    main_param_settings = {
        'dm41':    {'init': 0.87,  'lower': 0.01,  'upper': 100.,        'nsteps': 50, 'scale': 'log'},
        'dm51':    {'init': 0.47,  'lower': 0.01,  'upper': 100.,        'nsteps': 50, 'scale': 'log'},
        'theta14': {'init': 0.15,  'lower': 1e-4,  'upper': np.sqrt(0.9),'nsteps': 50, 'scale': 'log'},
        'theta15': {'init': 0.13,  'lower': 1e-4,  'upper': np.sqrt(0.9),'nsteps': 50, 'scale': 'log'},
        'theta24': {'init': 0.13,  'lower': 1e-4,  'upper': np.sqrt(0.9),'nsteps': 50, 'scale': 'log'},
        'theta25': {'init': 0.17,  'lower': 1e-4,  'upper': np.sqrt(0.9),'nsteps': 50, 'scale': 'log'},
        'theta34': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'theta35': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'theta45': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'delta14': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'delta15': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'delta24': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'delta25': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'delta34': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'delta35': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'delta45': {'init': 0.,    'lower': 0.,    'upper': 0.,           'nsteps': 0,  'scale': 'linear'},
        'sin2_2theta_eff': {'init': 1e-2, 'lower': 1e-4, 'upper': 1.0, 'nsteps': 50, 'scale': 'log'},
    }

    def build_axis(key):
        cfg = main_param_settings[key]
        lo, hi, n = float(cfg['lower']), float(cfg['upper']), int(cfg['nsteps'])
        if n <= 1:
            return np.array([float(cfg['init'])])
        return (np.logspace(np.log10(lo), np.log10(hi), n)
                if cfg['scale'] == 'log' else np.linspace(lo, hi, n))

    iv = {k: float(main_param_settings[k]['init']) for k in main_param_settings}

    # ── 初始化 ────────────────────────────────────────────────────
    exp           = MicroBooNE()
    model_wrapper = OscillationModelWrapper()
    likelihood    = LikelihoodCalculator(exp, model_wrapper, "1+3+1", main_param_settings)

    # ── chi2_null ─────────────────────────────────────────────────
    chi2_null = likelihood.compute_chi2_null()

    # ── 全局最小化 ────────────────────────────────────────────────
    global_best_params, chi2_global_min = likelihood.global_minimize(n_random=50, seed=42)
    delta_chi2_sig = chi2_null - chi2_global_min

    # ── Best-fit 结果输出 ─────────────────────────────────────────
    print("=" * 60)
    print("Best-Fit Results  (1+3+1 model, MiniBooNE)")
    print("=" * 60)
    print(f"  chi2_null        = {chi2_null:.4f}")
    print(f"  chi2_global_min  = {chi2_global_min:.4f}")
    print(f"  Delta_chi2       = {delta_chi2_sig:.4f}  "
          f"({np.sqrt(max(delta_chi2_sig, 0.)):.2f} sigma)")
    print("\n  Best-fit parameters:")
    key_params = ['theta14', 'theta24', 'theta15', 'theta25', 'dm41', 'dm51']
    for nm in key_params:
        val = global_best_params.get(nm, 0.)
        print(f"    {nm:12s} = {val:.6g}")

    # ── 扫描配置 ──────────────────────────────────────────────────
    scan_configs = {
        1: {'g1': build_axis('theta14'),
            'g2': build_axis('theta15'),
            'fb': {'dm41': iv['dm41'], 'dm51': iv['dm51']}},
        2: {'g1': build_axis('dm41'),
            'g2': build_axis('dm51'),
            'fb': {'theta14': iv['theta14'], 'theta24': iv['theta24'],
                   'theta15': iv['theta15'], 'theta25': iv['theta25']}},
        3: {'g1': build_axis('sin2_2theta_eff'),
            'g2': build_axis('dm41'),
            'fb': {'dm51': iv['dm51'],
                   'theta15': iv['theta15'], 'theta25': iv['theta25']}},
    }

    modes_to_run = list(range(1, 4)) if ALL_MODES else [PLOT_MODE]

    # ── 固定参数网格扫描 ──────────────────────────────────────────
    scan_results = {}
    for m in modes_to_run:
        cfg = scan_configs[m]
        g1, g2, fb = cfg['g1'], cfg['g2'], cfg['fb']
        chi2_grid, TS_grid, best_xy, chi2_scan_min, best_fp = \
            likelihood.scan_physical_2d(g1, g2, mode=m, fixed_base=fb,
                                        chi2_null_override=chi2_null)
        scan_results[m] = (chi2_grid, TS_grid, best_xy, chi2_scan_min, best_fp)

    # ── 扫描结果摘要 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Scan Results Summary")
    print("=" * 60)
    for m in modes_to_run:
        _, _, bxy, csm, _ = scan_results[m]
        print(f"  Mode {m}: chi2_scan_min={csm:.4f}  "
              f"Delta_chi2_sig={chi2_null - csm:.4f}  "
              f"best=({bxy[0]:.4g}, {bxy[1]:.4g})")

    # ── 绘图 ──────────────────────────────────────────────────────
    for m in modes_to_run:
        chi2_grid, TS_grid, best_xy, chi2_scan_min, _ = scan_results[m]
        g1, g2 = scan_configs[m]['g1'], scan_configs[m]['g2']

        global_best_xy = LikelihoodCalculator.params_to_plot_coords(global_best_params, m)
        ts_at_global   = _compute_ts_at_projection(
            g1, g2, chi2_grid, chi2_scan_min, global_best_xy,
            xscale='log', yscale='log')

        plot_2d_contour(
            grid1=g1, grid2=g2, TS_matrix=TS_grid, mode=m,
            best_xy=best_xy,
            global_best_xy=global_best_xy,
            ts_at_global_best=ts_at_global,
            chi2_null=chi2_null,
            chi2_scan_min=chi2_scan_min,
            chi2_global_min=chi2_global_min,
            model_label="MiniBooNE 1+3+1",
            savepath=f"miniboone_mode{m}.png",
        )


if __name__ == "__main__":
    main()