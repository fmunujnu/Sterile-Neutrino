import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 获取MiniBooNE反中微子模式通量数据
# ============================================================

class FluxData:
    """MiniBooNE通量数据管理类"""

    def __init__(self):
        self.flux_data = self._load_flux_data()
        self._process_flux_data()

    def _load_flux_data(self):
        """加载通量数据"""

        Elo = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])

        Ehi = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])

        numu_flux = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

        nue_flux = np.array([0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001])

        numub_flux = np.array([0.05, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0])

        nueb_flux = np.array([0.005, 0.004, 0.003, 0.002, 0.001, 0, 0, 0, 0, 0])

        return {
            'energy_bins_edges': (Elo, Ehi),
            'numu_flux': numu_flux,
            'numub_flux': numub_flux,
            'nue_flux': nue_flux,
            'nueb_flux': nueb_flux
        }

    def _process_flux_data(self):
        """处理通量数据"""
        Elo, Ehi = self.flux_data['energy_bins_edges']

        # 计算中点能量
        self.E_centers = (Elo + Ehi) / 2

        # 获取各味通量
        flux_arrays = {
            'numu': self.flux_data['numu_flux'],
            'numub': self.flux_data['numub_flux'],
            'nue': self.flux_data['nue_flux'],
            'nueb': self.flux_data['nueb_flux']
        }

        # 计算每个bin的总通量（原版方法）
        self.total_flux_per_bin = np.zeros_like(self.flux_data['numu_flux'])
        for flux_array in flux_arrays.values():
            self.total_flux_per_bin += flux_array

        # 计算总通量
        self.total_flux_sum = np.sum(self.total_flux_per_bin)
        self.total_four_flavors = sum(np.sum(f) for f in flux_arrays.values())

        # 计算归一化权重（原版方法）
        self.norm_weights = {}
        for flavor, flux_array in flux_arrays.items():
            self.norm_weights[flavor] = flux_array / self.total_four_flavors

    def get_energy_spectrum(self, flavor='numub'):
        """获取指定味的能量谱"""
        if flavor not in self.norm_weights:
            raise ValueError(f"无效的味类型: {flavor}。可选: {list(self.norm_weights.keys())}")

        return self.E_centers.copy(), self.norm_weights[flavor].copy()


# ============================================================
# 2. 两味旋转矩阵
# ============================================================

def rotation_matrix(N, i, j, theta, delta=0.0):
    """创建两味旋转矩阵"""
    R = np.identity(N, dtype=complex)
    R[i, i] = np.cos(theta)
    R[j, j] = np.cos(theta)
    R[i, j] = np.sin(theta) * np.exp(-1j * delta)
    R[j, i] = -np.sin(theta) * np.exp(1j * delta)
    return R


# ============================================================
# 3. 构造不同模型的混合矩阵
# ============================================================

class NeutrinoModels:
    """中微子振荡模型构建类"""

    def __init__(self):
        self.model_definitions = {
            "3": self._build_3flavor,
            "3+1": self._build_3plus1,
            "3+2": self._build_3plus2,
            "1+3+1": self._build_1plus3plus1
        }

    def _build_3flavor(self):
        """构建3味振荡模型"""
        N = 3
        U = (
                rotation_matrix(N, 1, 2, 49 * np.pi / 180) @
                rotation_matrix(N, 0, 2, 8.6 * np.pi / 180) @
                rotation_matrix(N, 0, 1, 33 * np.pi / 180)
        )
        m2 = np.array([0.0, 7.5e-5, 2.5e-3])
        return U, m2, N

    def _build_3plus1(self):
        """构建3+1振荡模型"""
        N = 4
        U = (
                rotation_matrix(N, 2, 3, np.arcsin(0)) @
                rotation_matrix(N, 1, 3, np.arcsin(0.17)) @
                rotation_matrix(N, 0, 3, np.arcsin(0.15)) @
                rotation_matrix(N, 1, 2, 49 * np.pi / 180) @
                rotation_matrix(N, 0, 2, 8.6 * np.pi / 180) @
                rotation_matrix(N, 0, 1, 33 * np.pi / 180)
        )
        m2 = np.array([0.0, 7.5e-5, 2.5e-3, 1])
        return U, m2, N

    def _build_3plus2(self):
        """构建3+2振荡模型"""
        N = 5
        U = (
                rotation_matrix(N, 3, 4, np.arcsin(0)) @
                rotation_matrix(N, 2, 4, np.arcsin(0)) @
                rotation_matrix(N, 1, 4, np.arcsin(0.13)) @
                rotation_matrix(N, 0, 4, np.arcsin(0.14)) @
                rotation_matrix(N, 2, 3, np.arcsin(0)) @
                rotation_matrix(N, 1, 3, np.arcsin(0.15)) @
                rotation_matrix(N, 0, 3, np.arcsin(0.13)) @
                rotation_matrix(N, 1, 2, 49 * np.pi / 180) @
                rotation_matrix(N, 0, 2, 8.6 * np.pi / 180) @
                rotation_matrix(N, 0, 1, 33 * np.pi / 180)
        )
        m2 = np.array([0.0, 7.5e-5, 2.5e-3, 0.47, 0.87])
        return U, m2, N

    def _build_1plus3plus1(self):
        """构建1+3+1振荡模型"""
        N = 5
        U = (
                rotation_matrix(N, 3, 4, np.arcsin(0)) @
                rotation_matrix(N, 2, 4, np.arcsin(0)) @
                rotation_matrix(N, 1, 4, np.arcsin(0.17)) @
                rotation_matrix(N, 0, 4, np.arcsin(0.13)) @
                rotation_matrix(N, 2, 3, np.arcsin(0)) @
                rotation_matrix(N, 1, 3, np.arcsin(0.13)) @
                rotation_matrix(N, 0, 3, np.arcsin(0.15)) @
                rotation_matrix(N, 1, 2, 49 * np.pi / 180) @
                rotation_matrix(N, 0, 2, 8.6 * np.pi / 180) @
                rotation_matrix(N, 0, 1, 33 * np.pi / 180)
        )
        m2 = np.array([0.0, 7.5e-5, 2.5e-3, -0.87, 0.47])
        return U, m2, N

    def build_model(self, model_name):
        """构建指定模型"""
        if model_name not in self.model_definitions:
            raise ValueError(f"未知模型: {model_name}")
        return self.model_definitions[model_name]()


# ============================================================
# 4. 能谱平均振荡概率
# ============================================================

class OscillationCalculator:
    """振荡概率计算器"""

    def __init__(self, flux_data, model_builder):
        self.flux = flux_data
        self.models = model_builder

    def calculate_average_probability(self, model_name, initial_flavor, target_idx, L_km):
        """计算平均振荡概率"""
        U, m2, N = self.models.build_model(model_name)
        E_centers, flux_weights = self.flux.get_energy_spectrum(initial_flavor)

        # 创建初始态分布向量
        flavor_to_index = {'numu': 1, 'numub': 1, 'nue': 0, 'nueb': 0}
        if initial_flavor not in flavor_to_index:
            raise ValueError(f"无效的初始味: {initial_flavor}")

        initial_dist = np.zeros(N, dtype=complex)
        initial_dist[flavor_to_index[initial_flavor]] = 1.0

        # 向量化计算
        prob = 0.0
        for E, w in zip(E_centers, flux_weights):
            phase = np.exp(-1j * 1.267 * m2 * L_km / E)
            amp = 0.0 + 0.0j
            for a in range(len(initial_dist)):
                if initial_dist[a] != 0:
                    amp += initial_dist[a] * np.sum(
                        U[target_idx, :] * np.conj(U[a, :]) * phase
                    )
            prob += w * np.abs(amp) ** 2
        return prob

    def compute_all_probabilities(self, L_range=(0.00, 0.5), num_points=500):
        """计算所有模型和情况的振荡概率"""
        L_km = np.linspace(L_range[0], L_range[1], num_points)
        L_m = L_km * 1000.0

        model_names = ["3", "3+1", "3+2", "1+3+1"]
        all_probs = {}

        for model_name in model_names:
            model_probs = {}

            # 计算所有振荡情况
            cases = [
                ('numu', 'nue', 0),  # νμ → νe
                ('numu', 'numu', 1),  # νμ → νμ
                ('numub', 'nueb', 0),  # 反νμ → 反νe
                ('numub', 'numub', 1),  # 反νμ → 反νμ
                ('nue', 'nue', 0),  # νe → νe (存活)
                ('nueb', 'nueb', 0),  # 反νe → 反νe (存活)
            ]

            for initial, target_name, target_idx in cases:
                probs = np.zeros_like(L_km)
                for i, L in enumerate(L_km):
                    probs[i] = self.calculate_average_probability(
                        model_name, initial, target_idx, L
                    )
                model_probs[(initial, target_name)] = probs

            all_probs[model_name] = model_probs

        return L_m, all_probs, model_names


# ============================================================
# 5. 绘图函数
# ============================================================

def plot_all_oscillation_probabilities_in_one():
    """在一张图中绘制六种振荡概率图，使用2行3列布局"""
    print("MiniBooNE反中微子模式振荡概率计算")
    print("=" * 60)

    # 初始化组件
    flux_data = FluxData()
    model_builder = NeutrinoModels()
    calculator = OscillationCalculator(flux_data, model_builder)

    # 计算所有概率
    print("正在计算振荡概率...")
    L_m, all_probs, models = calculator.compute_all_probabilities(
        L_range=(0.001, 0.5),
        num_points=500
    )

    print("计算完成！开始绘制六合一图片...")

    # 计算各味总通量
    total_numu_flux = np.sum(flux_data.flux_data['numu_flux'])
    total_numub_flux = np.sum(flux_data.flux_data['numub_flux'])
    total_nue_flux = np.sum(flux_data.flux_data['nue_flux'])
    total_nueb_flux = np.sum(flux_data.flux_data['nueb_flux'])
    total_all_flux = flux_data.total_flux_sum

    print(f"\n通量归一化信息:")
    print(f"νμ占总通量比例: {total_numu_flux / total_all_flux * 100:.4f}%")
    print(f"反νμ占总通量比例: {total_numub_flux / total_all_flux * 100:.4f}%")
    print(f"νe占总通量比例: {total_nue_flux / total_all_flux * 100:.4f}%")
    print(f"反νe占总通量比例: {total_nueb_flux / total_all_flux * 100:.4f}%")

    # 创建2行3列的子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 定义六种情况
    cases = [
        {
            'row': 0, 'col': 0,
            'target': 'nue',
            'title': r'$\nu_e$',
            'ylabel': r'$P(\nu_e)$ [%]'
        },
        {
            'row': 0, 'col': 1,
            'target': 'nueb',
            'title': r'$\bar{\nu}_e$',
            'ylabel': r'$P(\bar{\nu}_e)$ [%]'
        },
        {
            'row': 0, 'col': 2,
            'type': 'combined',
            'flavor': 'nue',
            'title': r'Total $\nu_e + \bar{\nu}_e$',
            'ylabel': r'$P(\nu_e+\bar{\nu}_e)$ [%]'
        },
        {
            'row': 1, 'col': 0,
            'target': 'numu',
            'title': r'$\nu_\mu$',
            'ylabel': r'$P(\nu_\mu)$ [%]'
        },
        {
            'row': 1, 'col': 1,
            'target': 'numub',
            'title': r'$\bar{\nu}_\mu$',
            'ylabel': r'$P(\bar{\nu}_\mu)$ [%]'
        },
        {
            'row': 1, 'col': 2,
            'type': 'combined',
            'flavor': 'numu',
            'title': r'Total $\nu_\mu + \bar{\nu}_\mu$',
            'ylabel': r'$P(\nu_\mu+\bar{\nu}_\mu)$ [%]'
        }
    ]

    # 为每个子图绘制数据
    for case in cases:
        ax = axes[case['row'], case['col']]

        if case.get('type') == 'combined':
            # 组合情况：νe + 反νe 或 νμ + 反νμ
            for model in models:
                model_probs = all_probs[model]

                if case['flavor'] == 'nue':
                    # 总电子中微子概率
                    total_probs = np.zeros_like(L_m)

                    # 从νμ到νe的贡献
                    if ('numu', 'nue') in model_probs:
                        total_probs += model_probs[('numu', 'nue')]

                    # 从νe到νe的贡献
                    if ('nue', 'nue') in model_probs:
                        total_probs += model_probs[('nue', 'nue')]

                    # 从反νμ到反νe的贡献
                    if ('numub', 'nueb') in model_probs:
                        total_probs += model_probs[('numub', 'nueb')]

                    # 从反νe到反νe的贡献
                    if ('nueb', 'nueb') in model_probs:
                        total_probs += model_probs[('nueb', 'nueb')]

                    ax.plot(L_m, total_probs * 100, linewidth=2, label=model)

                else:  # νμ + 反νμ
                    # 总μ子中微子概率
                    total_probs = np.zeros_like(L_m)

                    # 从νμ到νμ的贡献
                    if ('numu', 'numu') in model_probs:
                        total_probs += model_probs[('numu', 'numu')]

                    # 从反νμ到反νμ的贡献
                    if ('numub', 'numub') in model_probs:
                        total_probs += model_probs[('numub', 'numub')]

                    ax.plot(L_m, total_probs * 100, linewidth=2, label=model)
        else:
            # 单个味的情况
            for model in models:
                model_probs = all_probs[model]

                if case['target'] == 'nue':
                    # νe概率：来自νμ→νe和νe→νe
                    total_probs = np.zeros_like(L_m)

                    # 从νμ到νe的贡献
                    if ('numu', 'nue') in model_probs:
                        total_probs += model_probs[('numu', 'nue')]

                    # 从νe到νe的贡献
                    if ('nue', 'nue') in model_probs:
                        total_probs += model_probs[('nue', 'nue')]

                    ax.plot(L_m, total_probs * 100, linewidth=2, label=model)

                elif case['target'] == 'nueb':
                    # 反νe概率：来自反νμ→反νe和反νe→反νe
                    total_probs = np.zeros_like(L_m)

                    # 从反νμ到反νe的贡献
                    if ('numub', 'nueb') in model_probs:
                        total_probs += model_probs[('numub', 'nueb')]

                    # 从反νe到反νe的贡献
                    if ('nueb', 'nueb') in model_probs:
                        total_probs += model_probs[('nueb', 'nueb')]

                    ax.plot(L_m, total_probs * 100, linewidth=2, label=model)

                elif case['target'] == 'numu':
                    # νμ概率：主要来自νμ→νμ
                    total_probs = np.zeros_like(L_m)

                    # 从νμ到νμ的贡献
                    if ('numu', 'numu') in model_probs:
                        total_probs += model_probs[('numu', 'numu')]

                    ax.plot(L_m, total_probs * 100, linewidth=2, label=model)

                elif case['target'] == 'numub':
                    # 反νμ概率：主要来自反νμ→反νμ
                    total_probs = np.zeros_like(L_m)

                    # 从反νμ到反νμ的贡献
                    if ('numub', 'numub') in model_probs:
                        total_probs += model_probs[('numub', 'numub')]

                    ax.plot(L_m, total_probs * 100, linewidth=2, label=model)

        ax.set_xlabel("Baseline L (m)", fontsize=11)
        ax.set_ylabel(case['ylabel'], fontsize=11)
        ax.set_title(case['title'], fontsize=13)
        ax.grid(True, alpha=0.3)

    # 只在第一个子图显示图例
    axes[0, 0].legend(fontsize=10, loc='upper left')

    # 调整布局（还原为原始标题）
    plt.suptitle(
        r"Neutrino Oscillation Probabilities for Different Models (MicroBooNE Flux $\nu_\mu$ mode)",
        fontsize=16, y=0.98)
    plt.tight_layout()

    # 保存图片
    plt.savefig('all_oscillation_probabilities.png', dpi=150, bbox_inches='tight')
    print("已保存图片: all_oscillation_probabilities.png")

    # 显示图片
    plt.show()


# ============================================================
# 运行主程序
# ============================================================

if __name__ == "__main__":
    plot_all_oscillation_probabilities_in_one()