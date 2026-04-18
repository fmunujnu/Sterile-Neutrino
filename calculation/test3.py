import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
import matplotlib as mpl
import matplotlib.pyplot as plt
import read_data as rd

# 改microboone

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans'],
    'mathtext.fontset': 'dejavusans',
    'axes.unicode_minus': False,
    'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 11, 'figure.dpi': 100,
})


# ============================================================
# 实验数据
# ============================================================
class MicroBooNE:
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

        self.E_centers    = (self.Elo + self.Ehi) / 2.0
        self.n_flux_bins  = len(self.E_centers)
        self.absolute_flux = {'numu': self.numu_flux, 'numub': self.numub_flux,
                               'nue':  self.nue_flux,  'nueb':  self.nueb_flux}

        obs_Elo = np.linspace(0.0, 2.5, 26)
        obs_Ehi = np.r_[np.linspace(0.1, 2.5, 25), 10]

        self.obs_Elo_GeV     = obs_Elo
        self.obs_Ehi_GeV     = obs_Ehi
        self.obs_centers_GeV = (self.obs_Elo_GeV + self.obs_Ehi_GeV) / 2.
        self.n_obs_bins      = len(self.obs_centers_GeV)

        # ── 多通道结构 ──────────────────────────────────────────────
        # 通道 0：全部包含（FC）；通道 1：部分包含（PC）；通道 2–6：背景源
        # 信号仅出现在通道 0 和通道 1，各乘以对应的缩放系数 a / b
        self.n_channels         = 7
        self.signal_fractions   = np.array([0.7, 0.3, 0., 0., 0., 0., 0.])

        file1 = r"E:\sterile neutrino data\HEPData-ins3088922-v1-Unconstrained_14_channels.csv"

        data, bkg, sigbkg = rd.read_three_spectra(file1)

        data = data[0:182]
        bkg = bkg[0:182]

        self.observed = data
        self.background = bkg


        file2 = r"E:\sterile neutrino data\HEPData-ins3088922-v1-14_channel_covariance_matrix.csv"  # 改成你的文件路径

        cov = rd.read_covariance_matrix(
            file2,
            make_symmetric=False  # 如果只给了半个矩阵就设为True
        )

        cov = cov[:182, :182]

        self.covariance = np.array(cov, dtype=float)

        self.inv_covariance = inv(self.covariance)

###################################################################################################
        self.N_target = 1.3e30
        self.L_km = 0.470
        self.pot = 6.46e20
        self.sigma_flux = np.full(self.n_flux_bins, 0.93e-39)
        self.efficiency_flux = np.full(self.n_flux_bins, 1)

        self.E_true = self.E_centers
        self.E_true_edges = np.concatenate([[self.Elo[0]], self.Ehi])
        self.dE_true = np.diff(self.E_true_edges)

        self.response_matrix = np.zeros((self.n_flux_bins, self.n_obs_bins))
        for i, E in enumerate(self.E_centers):
            for j in range(self.n_obs_bins):
                if self.obs_Elo_GeV[j] <= E < self.obs_Ehi_GeV[j]:
                    self.response_matrix[i, j] = 1.0; break
        rs = self.response_matrix.sum(axis=1, keepdims=True)
        nz = rs[:, 0] > 0
        self.response_matrix[nz] /= rs[nz]

        scale = self.sigma_flux * self.efficiency_flux * self.pot * self.N_target
        self.precomp_nu  = self.absolute_flux['numu']  * scale
        self.precomp_nub = self.absolute_flux['numub'] * scale


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
                      "delta14","delta24","delta34","delta15","delta25","delta35","delta45","dm41","dm51"],
        }

    def get_mixing_and_masses(self, model_name, params):
        builders = {"3": self._build_3flavor, "3+1": self._build_3plus1,
                    "3+2": self._build_3plus2, "1+3+1": self._build_1plus3plus1}
        if model_name not in builders:
            raise ValueError(f"未知模型: {model_name}")
        return builders[model_name]() if model_name == "3" else builders[model_name](params)

    @staticmethod
    def _sa(v):
        return np.arcsin(np.clip(v, 0., 1.))

    def _U3(self, N):
        return (rotation_matrix(N, 1, 2, self.theta23) @
                rotation_matrix(N, 0, 2, self.theta13, delta=self.delta13) @
                rotation_matrix(N, 0, 1, self.theta12))

    def _build_3flavor(self):
        U = self._U3(3)
        return U, self.m2_3flavor.copy(), 3

    def _build_3plus1(self, p):
        N, sa = 4, self._sa
        U = (rotation_matrix(N, 2, 3, sa(p.get("theta34",0.)), delta=p.get("delta34",0.)) @
             rotation_matrix(N, 1, 3, sa(p.get("theta24",0.)), delta=p.get("delta24",0.)) @
             rotation_matrix(N, 0, 3, sa(p.get("theta14",0.)), delta=p.get("delta14",0.))) @ self._U3(N)
        return U, np.append(self.m2_3flavor, p.get("dm41", 1.0)), N

    def _build_3plus2(self, p):
        N, sa = 5, self._sa
        Us1 = (rotation_matrix(N, 2, 3, sa(p.get("theta34",0.)), delta=p.get("delta34",0.)) @
               rotation_matrix(N, 1, 3, sa(p.get("theta24",0.)), delta=p.get("delta24",0.)) @
               rotation_matrix(N, 0, 3, sa(p.get("theta14",0.)), delta=p.get("delta14",0.)))
        Us2 = (rotation_matrix(N, 3, 4, sa(p.get("theta45",0.)), delta=p.get("delta45",0.)) @
               rotation_matrix(N, 2, 4, sa(p.get("theta35",0.)), delta=p.get("delta35",0.)) @
               rotation_matrix(N, 1, 4, sa(p.get("theta25",0.)), delta=p.get("delta25",0.)) @
               rotation_matrix(N, 0, 4, sa(p.get("theta15",0.)), delta=p.get("delta15",0.)))
        U = Us2 @ Us1 @ self._U3(N)
        return U, np.append(self.m2_3flavor, [p.get("dm41",0.47), p.get("dm51",0.87)]), N

    def _build_1plus3plus1(self, p):
        N, sa = 5, self._sa
        Us1 = (rotation_matrix(N, 2, 3, sa(p.get("theta34",0.)), delta=p.get("delta34",0.)) @
               rotation_matrix(N, 1, 3, sa(p.get("theta24",0.)), delta=p.get("delta24",0.)) @
               rotation_matrix(N, 0, 3, sa(p.get("theta14",0.)), delta=p.get("delta14",0.)))
        Us2 = (rotation_matrix(N, 3, 4, sa(p.get("theta45",0.)), delta=p.get("delta45",0.)) @
               rotation_matrix(N, 2, 4, sa(p.get("theta35",0.)), delta=p.get("delta35",0.)) @
               rotation_matrix(N, 1, 4, sa(p.get("theta25",0.)), delta=p.get("delta25",0.)) @
               rotation_matrix(N, 0, 4, sa(p.get("theta15",0.)), delta=p.get("delta15",0.)))
        U = Us2 @ Us1 @ self._U3(N)
        return U, np.append(self.m2_3flavor, [p.get("dm41",-0.87), p.get("dm51",0.47)]), N


def oscillation_probability_array(U, m2, alpha, beta, L, E_arr, anti=False):
    phase  = np.exp(-1j * 1.267 * m2[np.newaxis, :] * L / E_arr[:, np.newaxis])
    coeffs = np.conj(U[beta, :]) * U[alpha, :] if anti else U[beta, :] * np.conj(U[alpha, :])
    return np.abs(phase @ coeffs) ** 2


# ============================================================
# 事件率计算器
# ============================================================
class EventRateCalculator:
    def __init__(self, exp, model):
        self.L_km            = exp.L_km
        self.E_true          = exp.E_true
        self.R               = exp.response_matrix
        self.pc_nu           = exp.precomp_nu
        self.pc_nub          = exp.precomp_nub
        self.model           = model
        # 多通道参数：通道数与各通道信号权重
        self.n_channels      = exp.n_channels
        self.signal_fractions = exp.signal_fractions

    def compute_signal_spectrum(self, model_name, params):
        """
        计算展平的多通道信号谱，形状为 (n_channels * n_obs_bins,)。

        步骤：
          1. 按原方式计算每个观测能量 bin 的信号数（形状 (n_obs_bins,)）。
          2. 依次将各通道的信号乘以对应的 signal_fractions 系数，
             通道 0（全部包含）和通道 1（部分包含）保留信号，
             通道 2–6（背景源通道）置零。
          3. 拼接为一维数组返回，长度 = n_channels * n_obs_bins。
        """
        U, m2, _ = self.model.get_mixing_and_masses(model_name, params)
        P_nu  = oscillation_probability_array(U, m2, 1, 0, self.L_km, self.E_true, False)
        P_nub = oscillation_probability_array(U, m2, 1, 0, self.L_km, self.E_true, True)
        # 26-bin 基础信号谱
        signal_26 = self.R.T @ (self.pc_nu * P_nu + self.pc_nub * P_nub)
        # 展平为 182-bin 多通道谱：每通道乘以对应权重
        signal_flat = np.concatenate(
            [signal_26 * frac for frac in self.signal_fractions]
        )
        return signal_flat


# ============================================================
# 似然计算器
# ============================================================
class LikelihoodCalculator:

    def __init__(self, exp, model, model_name, param_settings):
        self.exp         = exp
        self.model_name  = model_name
        self.param_names = model.sterile_param_names[model_name]
        self.param_info  = {}
        self.param_scale = {}

        for nm in self.param_names:
            if nm not in param_settings:
                raise KeyError(f"参数 '{nm}' 未在 param_settings 中定义。")
            cfg = dict(param_settings[nm])
            self.param_scale[nm] = cfg.get('scale', 'linear')
            self.param_info[nm]  = cfg

        self.rate_calc           = EventRateCalculator(exp, model)
        self._chi2_global_min    = None
        self._global_best_params = None

    # ── 参数空间变换 ──────────────────────────────────────────────
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
                lo, hi = np.log10(max(lo, 1e-12)), np.log10(max(hi, 1e-12))
            out.append((lo, hi))
        return out

    def get_initial_guess(self):
        return np.array([self._to_opt(self.param_info[nm]['init'], nm)
                         for nm in self.param_names])

    def get_init_param_dict(self):
        return {nm: float(self.param_info[nm]['init']) for nm in self.param_names}

    # ── 对数似然 ──────────────────────────────────────────────────
    def log_likelihood(self, x):
        p     = self._v2p(x)
        n_obs = self.exp.observed
        bkg   = self.exp.background
        S     = self.rate_calc.compute_signal_spectrum(self.model_name, p)
        mu    = np.maximum(S + bkg, 1e-12)
        cov_a = self.exp.covariance * np.outer(mu, mu)
        n     = cov_a.shape[0]
        cov_r = cov_a + 1e-8 * np.trace(cov_a) / n * np.eye(n)
        try:    inv_c = np.linalg.inv(cov_r)
        except: inv_c = np.linalg.pinv(cov_r)
        d = n_obs - mu
        return 0.5 * float(d @ inv_c @ d)

    def compute_chi2_null(self):
        bkg   = self.exp.background
        cov_a = self.exp.covariance * np.outer(bkg, bkg)
        n     = cov_a.shape[0]
        cov_r = cov_a + 1e-8 * np.trace(cov_a) / n * np.eye(n)
        try:    inv_c = np.linalg.inv(cov_r)
        except: inv_c = np.linalg.pinv(cov_r)
        d = self.exp.observed - bkg
        return float(d @ inv_c @ d)

    # ── 全局最小化 ────────────────────────────────────────────────
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
                    pt[k] = (rng.uniform(np.log10(lo_ph), np.log10(hi_ph))
                             if self.param_scale[nm] == 'log'
                             else rng.uniform(lo_ph, hi_ph))
            starts.append(np.clip(pt, [lo for lo, hi in bounds], [hi for lo, hi in bounds]))
        for _ in range(n_random):
            starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

        opt = dict(method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-9})
        best_x, best_c2 = None, np.inf
        for xs in starts:
            try:
                res = minimize(self.log_likelihood, xs, **opt)
                c2  = 2. * res.fun
                if c2 < best_c2:
                    best_c2, best_x = c2, res.x.copy()
            except Exception:
                continue

        if best_x is None:
            best_x = x0; best_c2 = 2. * self.log_likelihood(x0)

        try:
            res2 = minimize(self.log_likelihood, best_x, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if 2. * res2.fun < best_c2:
                best_c2, best_x = 2. * res2.fun, res2.x.copy()
        except Exception:
            pass

        best_params = self._v2p(best_x)
        self._chi2_global_min    = float(best_c2)
        self._global_best_params = best_params
        return best_params, float(best_c2)

    # ── 剖面似然（修正版） ────────────────────────────────────────
    def profile_likelihood(self, scan_fixed, param_roles=None, n_random_starts=5, seed=None):
        if param_roles is None:
            param_roles = {}

        bounds_all = self.get_bounds()

        fixed_vals = {}
        free_idx   = []
        free_nms   = []
        free_bds   = []

        for i, nm in enumerate(self.param_names):
            lo, hi = bounds_all[i]

            if nm in scan_fixed:
                fixed_vals[nm] = self._to_opt(scan_fixed[nm], nm)

            else:
                role = param_roles.get(nm, 'fixed_init')

                if role == 'free':
                    free_idx.append(i)
                    free_nms.append(nm)
                    free_bds.append((lo, hi))

                elif role == 'fixed_init':
                    fixed_vals[nm] = self._to_opt(self.param_info[nm]['init'], nm)

                else:
                    try:
                        val = float(role)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"param_roles['{nm}'] = {role!r} 无效，"
                            f"请使用 'free'、'fixed_init' 或一个数值。")
                    fixed_vals[nm] = self._to_opt(val, nm)

        if not free_idx:
            full_x = np.array([
                fixed_vals[nm] for nm in self.param_names
            ])
            val = self.log_likelihood(full_x)
            return self._v2p(full_x), -val

        def objective(free_x):
            full_x = np.empty(len(self.param_names))
            for nm, fv in fixed_vals.items():
                full_x[self.param_names.index(nm)] = fv
            for k, idx in enumerate(free_idx):
                full_x[idx] = free_x[k]
            return self.log_likelihood(full_x)

        rng = np.random.default_rng(seed)

        starts = []
        if self._global_best_params is not None:
            gx = np.array([
                np.clip(
                    self._to_opt(self._global_best_params.get(nm, self.param_info[nm]['init']), nm),
                    free_bds[k][0], free_bds[k][1]
                )
                for k, nm in enumerate(free_nms)
            ])
            starts.append(gx)

        x0_free = np.array([
            np.clip(self._to_opt(self.param_info[nm]['init'], nm), lo, hi)
            for nm, (lo, hi) in zip(free_nms, free_bds)
        ])
        starts.append(x0_free)

        for _ in range(5):
            noise = rng.normal(0., 0.3, size=x0_free.shape)
            starts.append(np.clip(x0_free + noise,
                                  [lo for lo, hi in free_bds],
                                  [hi for lo, hi in free_bds]))
        for _ in range(n_random_starts):
            starts.append(np.array([rng.uniform(lo, hi) for lo, hi in free_bds]))

        opt_kwargs = dict(method='L-BFGS-B', bounds=free_bds,
                          options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-9})

        best_fun, best_free_x = np.inf, starts[0]
        for xs in starts:
            try:
                res = minimize(objective, xs, **opt_kwargs)
                if res.fun < best_fun:
                    best_fun   = res.fun
                    best_free_x = res.x.copy()
            except Exception:
                continue

        try:
            res2 = minimize(objective, best_free_x, method='L-BFGS-B', bounds=free_bds,
                            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res2.fun < best_fun:
                best_fun    = res2.fun
                best_free_x = res2.x.copy()
        except Exception:
            pass

        full_x = np.empty(len(self.param_names))
        for nm, fv in fixed_vals.items():
            full_x[self.param_names.index(nm)] = fv
        for k, idx in enumerate(free_idx):
            full_x[idx] = best_free_x[k]

        return self._v2p(full_x), -best_fun

    # ── 2D 固定参数网格扫描 ───────────────────────────────────────
    def scan_2d_fixed(self, scan_axis_1_name, grid1,
                      scan_axis_2_name, grid2,
                      extra_fixed=None,
                      coupled_params=None):
        if extra_fixed is None:
            extra_fixed = {}

        n1, n2    = len(grid1), len(grid2)
        chi2_grid = np.full((n1, n2), np.nan)
        param_store = [[None] * n2 for _ in range(n1)]

        base_params = self.get_init_param_dict()
        for k, v in extra_fixed.items():
            if k in base_params:
                base_params[k] = float(v)

        for i, v1 in enumerate(grid1):
            for j, v2 in enumerate(grid2):
                p = dict(base_params)
                p[scan_axis_1_name] = float(v1)
                p[scan_axis_2_name] = float(v2)
                if coupled_params is not None:
                    coupled_params(p, float(v1), float(v2))
                x = np.array([self._to_opt(p[nm], nm) for nm in self.param_names])
                chi2_grid[i, j]   = 2. * self.log_likelihood(x)
                param_store[i][j] = dict(p)

        chi2_scan_min = float(np.nanmin(chi2_grid))
        TS_grid       = np.maximum(0., chi2_grid - chi2_scan_min)
        assert float(np.nanmin(TS_grid)) < 1e-6, \
            "TS_grid 最小值不为零，chi2_scan_min 计算可能有误。"

        idx_best         = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
        best_full_params = param_store[idx_best[0]][idx_best[1]]
        best_xy          = (float(grid1[idx_best[0]]), float(grid2[idx_best[1]]))
        return chi2_grid, TS_grid, best_xy, chi2_scan_min, best_full_params

    # ── 2D 剖面似然扫描 ───────────────────────────────────────────
    def scan_2d_profile(self, scan_axis_1_name, grid1,
                        scan_axis_2_name, grid2,
                        param_roles=None,
                        coupled_params=None,
                        n_random_starts=5):
        assert self._chi2_global_min is not None, \
            "请先调用 global_minimize() 再运行 scan_2d_profile()。"
        if param_roles is None:
            param_roles = {}

        n1, n2    = len(grid1), len(grid2)
        chi2_grid = np.full((n1, n2), np.nan)
        param_store = [[None] * n2 for _ in range(n1)]
        total = n1 * n2
        done  = 0

        for i, v1 in enumerate(grid1):
            for j, v2 in enumerate(grid2):
                scan_fixed = {
                    scan_axis_1_name: float(v1),
                    scan_axis_2_name: float(v2),
                }
                if coupled_params is not None:
                    coupled_params(scan_fixed, float(v1), float(v2))

                best_p, neg_logL = self.profile_likelihood(
                    scan_fixed, param_roles=param_roles,
                    n_random_starts=n_random_starts, seed=done
                )
                chi2_grid[i, j]   = -2. * neg_logL
                param_store[i][j] = dict(best_p)
                done += 1
                if done % max(1, total // 20) == 0:
                    print(f"  [profile scan] {done}/{total} ({100*done/total:.0f}%)", flush=True)

        TS_grid = np.maximum(0., chi2_grid - self._chi2_global_min)

        idx_best         = np.unravel_index(np.nanargmin(chi2_grid), chi2_grid.shape)
        best_full_params = param_store[idx_best[0]][idx_best[1]]
        best_xy          = (float(grid1[idx_best[0]]), float(grid2[idx_best[1]]))
        return chi2_grid, TS_grid, best_xy, self._chi2_global_min, best_full_params

    @staticmethod
    def params_to_plot_coords(params, ax1_name, ax2_name):
        return params.get(ax1_name, np.nan), params.get(ax2_name, np.nan)


# ============================================================
# 全局 best-fit 投影点 TS 计算
# ============================================================
def _compute_ts_at_projection(g1, g2, chi2_grid, chi2_ref, proj_xy,
                               xscale='log', yscale='log'):
    px, py = proj_xy
    if not (px and py and np.isfinite(px) and np.isfinite(py) and px > 0 and py > 0):
        return np.inf
    tol = 1.05
    for val, g, scale in [(px, g1, xscale), (py, g2, yscale)]:
        if scale == 'log':
            if val < g[0]/tol or val > g[-1]*tol: return np.inf
        else:
            rng = g[-1] - g[0]
            if val < g[0] - 0.05*rng or val > g[-1] + 0.05*rng: return np.inf
    di = (int(np.argmin(np.abs(np.log10(g1) - np.log10(px))))
          if xscale == 'log' else int(np.argmin(np.abs(g1 - px))))
    dj = (int(np.argmin(np.abs(np.log10(g2) - np.log10(py))))
          if yscale == 'log' else int(np.argmin(np.abs(g2 - py))))
    c2 = chi2_grid[di, dj]
    return np.inf if np.isnan(c2) else float(max(0., c2 - chi2_ref))


# ============================================================
# bkg + 预测值 vs 观测数据 对比图
# ============================================================
def plot_bkg_pred_vs_data(exp, rate_calc, model_name, params,
                          param_label=None, savepath=None):
    """
    绘制"本底 + 模型预测信号"与观测数据的对比折线图（多通道汇总视图）。

    MicroBooNE 多通道处理策略：
      - background（绘图用）：所有 7 个通道的本底数据之和，每能量 bin 汇总。
      - observed（绘图用）：通道 0（全部包含）+ 通道 1（部分包含）之和，
        即信号敏感通道的观测事件数。
      - signal（绘图用）：同样只取通道 0 + 通道 1 的预测信号之和。
      - total_pred：上述 background + signal。

    数据排列约定：flat 数组按通道优先存储，即
      [ch0_e0…ch0_e25, ch1_e0…ch1_e25, …, ch6_e0…ch6_e25]，
    reshape 为 (n_channels=7, n_obs_bins=26)。

    Parameters
    ----------
    exp         : MicroBooNE 实验对象。
    rate_calc   : EventRateCalculator 对象。
    model_name  : str，模型名称（如 "1+3+1"）。
    params      : dict，模型参数字典。
    param_label : str or None，显示在标题中的参数描述；None 时自动生成。
    savepath    : str or None，非 None 时将图保存到该路径。

    Returns
    -------
    fig, ax : matplotlib Figure 与 Axes 对象。
    """
    n_ch = exp.n_channels   # 7
    n_E  = exp.n_obs_bins   # 26

    # ── 将 flat(182) 数组 reshape 为 (n_channels, n_obs_bins) ────
    bkg_mat    = exp.background.reshape(n_ch, n_E)
    data_mat   = exp.observed.reshape(n_ch, n_E)
    signal_flat = rate_calc.compute_signal_spectrum(model_name, params)
    signal_mat  = signal_flat.reshape(n_ch, n_E)

    # ── 各分量汇总 ────────────────────────────────────────────────
    # 背景：所有 7 个通道求和（反映总本底水平）
    bkg_sum    = bkg_mat.sum(axis=0)
    # 信号敏感通道（0=FC, 1=PC）：仅取前两通道
    signal_sum = signal_mat[0:2].sum(axis=0)
    data_sum   = data_mat[0:2].sum(axis=0)
    total_pred = bkg_sum + signal_sum

    energies  = exp.obs_centers_GeV
    xerr_lo   = energies - exp.obs_Elo_GeV
    xerr_hi   = exp.obs_Ehi_GeV - energies

    # ── 统计误差（泊松近似：sqrt(N)） ────────────────────────────
    data_err = np.sqrt(np.maximum(data_sum, 1.))
    pred_err = np.sqrt(np.maximum(total_pred, 1.))

    # ── 自动生成参数标签 ──────────────────────────────────────────
    if param_label is None:
        key_params = ['theta14', 'theta24', 'theta15', 'theta25', 'dm41', 'dm51']
        parts = []
        for k in key_params:
            if k in params:
                parts.append(f"{k}={params[k]:.3g}")
        param_label = ',  '.join(parts) if parts else str(params)

    # ── 绘图 ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- 背景（所有通道之和，虚线，用于参考） ---
    ax.step(
        np.append(exp.obs_Elo_GeV, exp.obs_Ehi_GeV[-1]),
        np.append(bkg_sum, bkg_sum[-1]),
        where='post',
        color='steelblue', linestyle='--', linewidth=1.6,
        label='Background (all channels sum)'
    )

    # --- bkg + 预测（实线 + 误差阴影） ---
    E_step  = np.append(exp.obs_Elo_GeV, exp.obs_Ehi_GeV[-1])
    P_step  = np.append(total_pred, total_pred[-1])
    Pe_step = np.append(pred_err,   pred_err[-1])
    ax.step(E_step, P_step, where='post',
            color='steelblue', linestyle='-', linewidth=2.0,
            label='Bkg (all ch.) + signal (FC+PC)')
    ax.fill_between(E_step, P_step - Pe_step, P_step + Pe_step,
                    step='post', color='steelblue', alpha=0.18,
                    label=r'Prediction $\pm\sqrt{N}$')

    # --- 观测数据点（FC + PC 通道之和） ---
    ax.errorbar(
        energies, data_sum,
        yerr=data_err,
        xerr=[xerr_lo, xerr_hi],
        fmt='o', color='crimson', markersize=7,
        capsize=4, elinewidth=1.5, markeredgewidth=1.2,
        label='Observed data (FC + PC channels)'
    )

    # ── 文字标注 ─────────────────────────────────────────────────
    total_signal = float(np.sum(signal_sum))
    total_bkg    = float(np.sum(bkg_sum))
    total_data   = float(np.sum(data_sum))
    ax.text(
        0.97, 0.97,
        (f"Total bkg (all ch.) = {total_bkg:.1f}\n"
         f"Total signal (FC+PC) = {total_signal:.2f}\n"
         f"Total pred          = {total_bkg + total_signal:.1f}\n"
         f"Total data (FC+PC)  = {total_data:.0f}"),
        transform=ax.transAxes, fontsize=11,
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.85)
    )

    # ── 标签、标题、图例 ──────────────────────────────────────────
    ax.set_xlabel(r'Reconstructed $E_\nu^{\rm QE}\ \rm [GeV]$', fontsize=13)
    ax.set_ylabel('Event count', fontsize=13)
    ax.set_title(
        f'MicroBooNE {model_name} — Bkg + Prediction vs Data\n'
        + r'$\bf{Params:}$ ' + param_label,
        fontsize=12
    )
    ax.legend(fontsize=11, loc='lower left', framealpha=0.85)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_xlim(exp.obs_Elo_GeV[0] * 0.98, exp.obs_Ehi_GeV[-1] * 1.02)
    ax.set_ylim(bottom=0.)

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"  [plot_bkg_pred_vs_data] 图像已保存至: {savepath}")

    plt.show()
    return fig, ax


# ============================================================
# 绘图函数
# ============================================================
def plot_2d_contour(grid1, grid2, TS_matrix,
                    xlabel, ylabel, title,
                    xscale='log', yscale='log',
                    best_xy=None, global_best_xy=None, ts_at_global_best=None,
                    chi2_null=None, chi2_scan_min=None, chi2_global_min=None,
                    model_label="MicroBooNE 1+3+1", savepath=None):
    from matplotlib.lines import Line2D

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
    pcm = ax1.pcolormesh(X, Y, TS_matrix, shading='auto', cmap='viridis_r', vmin=0., vmax=vmax_h)
    cb  = fig.colorbar(pcm, cax=cax1)
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

    for ax in (ax1, ax2):
        ax.set_xscale(xscale); ax.set_yscale(yscale)
        ax.set_xlabel(xlabel, fontsize=14); ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.25)
    ax1.set_title(f"{model_label} — {title}  " + r"$\Delta\chi^2$ map", fontsize=12, pad=8)
    ax2.set_title(f"{model_label} — {title}  Allowed regions", fontsize=12, pad=8)

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
                       markeredgecolor='k', markeredgewidth=0.8, zorder=7, label='Best fit (TS≡0)')
        legend_handles.append(pt)

    show_global = (global_best_xy is not None and ts_at_global_best is not None and
                   np.isfinite(ts_at_global_best) and ts_at_global_best < chi2_dist.ppf(0.90, 2))
    if show_global:
        gpt, = ax2.plot(global_best_xy[0], global_best_xy[1], 'D', color='cyan', markersize=9,
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

    ax2.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=0.85)
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 主程序
# ============================================================

SCAN_METHOD    = 'fixed'
RUN_GLOBAL_FIT = True

def _coupled_mode1(p, v1, v2):
    s14 = float(np.clip(np.sqrt(max(v1, 0.)), 0., 1.))
    s15 = float(np.clip(np.sqrt(max(v2, 0.)), 0., 1.))
    p['theta14'] = s14;  p['theta24'] = s14
    p['theta15'] = s15;  p['theta25'] = s15

def _coupled_mode3(p, v1, v2):
    st = float(np.clip((np.clip(v1, 1e-9, 1.) / 4.) ** 0.25, 0., 1.))
    p['theta14'] = st;  p['theta24'] = st;  p['dm41'] = float(v2)

def _coupled_mode1_profile(sf, v1, v2):
    s14 = float(np.clip(np.sqrt(max(v1, 0.)), 0., 1.))
    s15 = float(np.clip(np.sqrt(max(v2, 0.)), 0., 1.))
    sf['theta14'] = s14;  sf['theta24'] = s14
    sf['theta15'] = s15;  sf['theta25'] = s15

def _coupled_mode3_profile(sf, v1, v2):
    st = float(np.clip((np.clip(v1, 1e-9, 1.) / 4.) ** 0.25, 0., 1.))
    sf['theta14'] = st;  sf['theta24'] = st;  sf['dm41'] = float(v2)


SCAN_TASKS = [
    {
        'axis1'  : '_prod14',
        'axis2'  : '_prod15',
        'grid1'  : np.logspace(np.log10(1e-4), np.log10(np.sqrt(0.9)), 50),
        'grid2'  : np.logspace(np.log10(1e-4), np.log10(np.sqrt(0.9)), 50),
        'xlabel' : r'$\sin\theta_{14}\cdot\sin\theta_{24}$',
        'ylabel' : r'$\sin\theta_{15}\cdot\sin\theta_{25}$',
        'title'  : r'$\sin\theta_{14}\sin\theta_{24}$ vs $\sin\theta_{15}\sin\theta_{25}$',
        'xscale' : 'log', 'yscale' : 'log',
        'savepath': 'microboone_mode1.png',
        'extra_fixed'           : {'dm41': 0.87, 'dm51': 0.47},
        'coupled_params_fixed'  : _coupled_mode1,
        'param_roles'           : {
            'dm41': 'fixed_init', 'dm51': 'fixed_init',
            'theta34': 0., 'theta35': 0., 'theta45': 0.,
            'delta14': 0., 'delta15': 0., 'delta24': 0.,
            'delta25': 0., 'delta34': 0., 'delta35': 0., 'delta45': 0.,
        },
        'coupled_params_profile': _coupled_mode1_profile,
        'global_best_xy_fn': lambda p: (
            p.get('theta14', 0.) * p.get('theta24', 0.),
            p.get('theta15', 0.) * p.get('theta25', 0.)
        ),
    },
    {
        'axis1'  : 'dm41',
        'grid1'  : np.logspace(np.log10(0.01), np.log10(100.), 50),
        'axis2'  : 'dm51',
        'grid2'  : np.logspace(np.log10(0.01), np.log10(100.), 50),
        'xlabel' : r'$|\Delta m^2_{41}|\ \rm[eV^2]$',
        'ylabel' : r'$|\Delta m^2_{51}|\ \rm[eV^2]$',
        'title'  : r'$|\Delta m^2_{41}|$ vs $|\Delta m^2_{51}|$',
        'xscale' : 'log', 'yscale' : 'log',
        'savepath': 'microboone_mode2.png',
        'extra_fixed'           : {
            'theta14': 0.15, 'theta24': 0.13,
            'theta15': 0.13, 'theta25': 0.17,
        },
        'coupled_params_fixed'  : None,
        'param_roles'           : {
            'theta14': 'free', 'theta24': 'free',
            'theta15': 'free', 'theta25': 'free',
            'theta34': 0., 'theta35': 0., 'theta45': 0.,
            'delta14': 0., 'delta15': 0., 'delta24': 0.,
            'delta25': 0., 'delta34': 0., 'delta35': 0., 'delta45': 0.,
        },
        'coupled_params_profile': None,
        'global_best_xy_fn': lambda p: (
            abs(p.get('dm41', 0.)),
            abs(p.get('dm51', 0.))
        ),
    },
    {
        'axis1'  : '_sin2_2theta_eff',
        'axis2'  : 'dm41',
        'grid1'  : np.logspace(np.log10(1e-4), np.log10(1.0), 100),
        'grid2'  : np.logspace(np.log10(0.01),  np.log10(100.), 100),
        'xlabel' : r'$\sin^2(2\theta_{\mu e}^{\rm eff})$',
        'ylabel' : r'$|\Delta m^2_{41}|\ \rm[eV^2]$',
        'title'  : r'$\sin^2(2\theta_{\mu e}^{\rm eff})$ vs $|\Delta m^2_{41}|$',
        'xscale' : 'log', 'yscale' : 'log',
        'savepath': 'microboone_mode3.png',
        'extra_fixed'           : {
            'dm51': 0.47, 'theta15': 0.13, 'theta25': 0.17,
        },
        'coupled_params_fixed'  : _coupled_mode3,
        'param_roles'           : {
            'dm51'   : 'fixed_init',
            'theta15': 'fixed_init',
            'theta25': 'fixed_init',
            'theta34': 0., 'theta35': 0., 'theta45': 0.,
            'delta14': 0., 'delta15': 0., 'delta24': 0.,
            'delta25': 0., 'delta34': 0., 'delta35': 0., 'delta45': 0.,
        },
        'coupled_params_profile': _coupled_mode3_profile,
        'global_best_xy_fn': lambda p: (
            4. * p.get('theta14', 0.)**2 * p.get('theta24', 0.)**2,
            abs(p.get('dm41', 0.))
        ),
    },
]

PARAM_SETTINGS = {
    'dm41'   : {'init': 0.87,  'lower': 0.01,  'upper': 100.,         'nsteps': 100, 'scale': 'log'},
    'dm51'   : {'init': 0.47,  'lower': 0.01,  'upper': 100.,         'nsteps': 50,  'scale': 'log'},
    'theta14': {'init': 0.15,  'lower': 1e-4,  'upper': np.sqrt(0.9), 'nsteps': 50,  'scale': 'log'},
    'theta15': {'init': 0.13,  'lower': 1e-4,  'upper': np.sqrt(0.9), 'nsteps': 50,  'scale': 'log'},
    'theta24': {'init': 0.13,  'lower': 1e-4,  'upper': np.sqrt(0.9), 'nsteps': 50,  'scale': 'log'},
    'theta25': {'init': 0.17,  'lower': 1e-4,  'upper': np.sqrt(0.9), 'nsteps': 50,  'scale': 'log'},
    'theta34': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'theta35': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'theta45': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'delta14': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'delta15': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'delta24': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'delta25': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'delta34': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'delta35': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'delta45': {'init': 0.,    'lower': 0.,    'upper': 0.,            'nsteps': 0,   'scale': 'linear'},
    'sin2_2theta_eff': {'init': 1e-2, 'lower': 1e-4, 'upper': 1.0, 'nsteps': 100, 'scale': 'log'},
}


def main():
    run_global = RUN_GLOBAL_FIT or (SCAN_METHOD == 'profile')

    # ── 初始化 ────────────────────────────────────────────────────
    exp        = MicroBooNE()
    model      = NeutrinoModels()
    likelihood = LikelihoodCalculator(exp, model, "1+3+1", PARAM_SETTINGS)

    # ── chi2_null ──────────────────────────────────────────────────
    chi2_null = likelihood.compute_chi2_null()
    print(f"\nchi2_null = {chi2_null:.4f}")

    # ── 全局最优拟合 ──────────────────────────────────────────────
    chi2_global_min    = None
    global_best_params = None

    if run_global:
        print("\n运行全局最优拟合（n_random=50）...")
        global_best_params, chi2_global_min = likelihood.global_minimize(n_random=50, seed=42)
        delta_chi2_sig = chi2_null - chi2_global_min

        print("=" * 60)
        print("Best-Fit Results  (1+3+1 model, MicroBooNE)")
        print("=" * 60)
        print(f"  chi2_null        = {chi2_null:.4f}")
        print(f"  chi2_global_min  = {chi2_global_min:.4f}")
        print(f"  Delta_chi2       = {delta_chi2_sig:.4f}  "
              f"({np.sqrt(max(delta_chi2_sig, 0.)):.2f} sigma)")
        print("\n  Best-fit parameters:")
        for nm in ['theta14','theta24','theta15','theta25','dm41','dm51']:
            print(f"    {nm:12s} = {global_best_params.get(nm, 0.):.6g}")

    # ── bkg + 预测 vs 数据 对比图 ─────────────────────────────────
    #
    # 支持两种使用方式：
    #   1. 全局最优参数点（run_global=True 时自动使用）
    #   2. 手动指定任意参数点（修改 custom_params 字典即可）
    #
    # 方式1：使用全局最优拟合参数
    if global_best_params is not None:
        print("\n绘制全局最优参数点的 bkg + 预测 vs 数据...")
        plot_bkg_pred_vs_data(
            exp        = exp,
            rate_calc  = likelihood.rate_calc,
            model_name = "1+3+1",
            params     = global_best_params,
            param_label= "Global best-fit",
            savepath   = "microboone_spectrum_bestfit.png",
        )

    # 方式2：手动指定任意参数点（取消注释并修改 custom_params 后使用）
    # custom_params = {
    #     'theta14': 0.15, 'theta24': 0.13,
    #     'theta15': 0.13, 'theta25': 0.17,
    #     'theta34': 0.,   'theta35': 0.,   'theta45': 0.,
    #     'delta14': 0.,   'delta15': 0.,   'delta24': 0.,
    #     'delta25': 0.,   'delta34': 0.,   'delta35': 0., 'delta45': 0.,
    #     'dm41'   : 0.87, 'dm51'   : 0.47,
    # }
    # plot_bkg_pred_vs_data(
    #     exp        = exp,
    #     rate_calc  = likelihood.rate_calc,
    #     model_name = "1+3+1",
    #     params     = custom_params,
    #     savepath   = "microboone_spectrum_custom.png",
    # )

    # ── 执行扫描任务 ──────────────────────────────────────────────
    print(f"\n扫描方法: {SCAN_METHOD.upper()}")
    print("=" * 60)

    for task_idx, task in enumerate(SCAN_TASKS):
        ax1_name  = task['axis1']
        ax2_name  = task['axis2']
        grid1     = task['grid1']
        grid2     = task['grid2']
        xlabel    = task['xlabel']
        ylabel    = task['ylabel']
        title     = task['title']
        xscale    = task.get('xscale', 'log')
        yscale    = task.get('yscale', 'log')
        savepath  = task.get('savepath', f'scan_task{task_idx+1}.png')

        print(f"\n[任务 {task_idx+1}] {ax1_name} vs {ax2_name}  "
              f"(网格: {len(grid1)}×{len(grid2)})")

        if SCAN_METHOD == 'fixed':
            extra_fixed          = task.get('extra_fixed', {})
            coupled_params_fixed = task.get('coupled_params_fixed', None)
            chi2_grid, TS_grid, best_xy, chi2_scan_min, best_fp = \
                likelihood.scan_2d_fixed(
                    ax1_name, grid1, ax2_name, grid2,
                    extra_fixed=extra_fixed,
                    coupled_params=coupled_params_fixed,
                )
            chi2_ref_for_ts = chi2_scan_min

        elif SCAN_METHOD == 'profile':
            param_roles              = task.get('param_roles', {})
            coupled_params_profile   = task.get('coupled_params_profile', None)
            chi2_grid, TS_grid, best_xy, chi2_ref_for_ts, best_fp = \
                likelihood.scan_2d_profile(
                    ax1_name, grid1, ax2_name, grid2,
                    param_roles=param_roles,
                    coupled_params=coupled_params_profile,
                    n_random_starts=5,
                )
            chi2_scan_min = float(np.nanmin(chi2_grid))

        else:
            raise ValueError(f"未知的 SCAN_METHOD='{SCAN_METHOD}'，请使用 'fixed' 或 'profile'。")

        print(f"  chi2_scan_min = {chi2_scan_min:.4f}  "
              f"Delta_chi2_sig = {chi2_null - chi2_scan_min:.4f}  "
              f"best = ({best_xy[0]:.4g}, {best_xy[1]:.4g})")

        global_best_xy    = None
        ts_at_global_best = None
        if global_best_params is not None:
            proj_fn = task.get('global_best_xy_fn', None)
            if proj_fn is not None:
                global_best_xy = proj_fn(global_best_params)
            else:
                global_best_xy = (
                    global_best_params.get(ax1_name, np.nan),
                    global_best_params.get(ax2_name, np.nan),
                )
            ts_at_global_best = _compute_ts_at_projection(
                grid1, grid2, chi2_grid, chi2_ref_for_ts, global_best_xy,
                xscale=xscale, yscale=yscale,
            )

        plot_2d_contour(
            grid1=grid1, grid2=grid2, TS_matrix=TS_grid,
            xlabel=xlabel, ylabel=ylabel, title=title,
            xscale=xscale, yscale=yscale,
            best_xy=best_xy,
            global_best_xy=global_best_xy,
            ts_at_global_best=ts_at_global_best,
            chi2_null=chi2_null,
            chi2_scan_min=chi2_scan_min,
            chi2_global_min=chi2_global_min,
            model_label="MicroBooNE 1+3+1",
            savepath=savepath,
        )


if __name__ == "__main__":
    main()