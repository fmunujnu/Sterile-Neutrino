import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 获取MiniBooNE反中微子模式通量数据
# ============================================================

class FluxData:
    """MiniBooNE通量数据管理类，支持双源处理"""

    def __init__(self, source_type='antinu_mode'):
        """
        初始化通量数据
        source_type:
            - 'antinu_mode': 反中微子模式 (原代码)
            - 'nu_mode': 中微子模式
            - 'combined': 两者合并
        """
        self.source_type = source_type
        self.flux_data_source1 = self._load_flux_data_source1()  # 源1
        self.flux_data_source2 = self._load_flux_data_source2()  # 源2
        self._process_flux_data()

    def _load_flux_data_source1(self):
        """加载源1的通量数据 (反中微子模式)"""
        # 能量区间边界 (GeV) - 每50 MeV一个bin
        Elo = np.array([1])  # 这里放你的实际数据
        Ehi = np.array([1])  # 这里放你的实际数据

        # 各味的通量 (numu/cm^2/proton-on-target/50 MeV)
        numu_flux = np.array([1])  # 这里放你的实际数据
        numub_flux = np.array([1])  # 这里放你的实际数据
        nue_flux = np.array([1])  # 这里放你的实际数据
        nueb_flux = np.array([1])  # 这里放你的实际数据

        return {
            'energy_bins_edges': (Elo, Ehi),
            'numu_flux': numu_flux[42:],
            'numub_flux': numub_flux[42:],
            'nue_flux': nue_flux[42:],
            'nueb_flux': nueb_flux[42:]
        }

    def _load_flux_data_source2(self):
        """加载源2的通量数据 (中微子模式)"""
        # 能量区间边界 (GeV) - 每50 MeV一个bin
        Elo = np.array([1])  # 这里放你的实际数据，可以与源1不同
        Ehi = np.array([1])  # 这里放你的实际数据，可以与源1不同

        # 各味的通量 (numu/cm^2/proton-on-target/50 MeV)
        # 源2的味分布可以与源1不同
        numu_flux = np.array([1])  # 这里放你的实际数据
        numub_flux = np.array([1])  # 这里放你的实际数据
        nue_flux = np.array([1])  # 这里放你的实际数据
        nueb_flux = np.array([1])  # 这里放你的实际数据

        return {
            'energy_bins_edges': (Elo, Ehi),
            'numu_flux': numu_flux[42:],
            'numub_flux': numub_flux[42:],
            'nue_flux': nue_flux[42:],
            'nueb_flux': nueb_flux[42:]
        }

    def _process_flux_data(self):
        """处理双源通量数据"""
        # 源1数据处理
        Elo1, Ehi1 = self.flux_data_source1['energy_bins_edges']
        self.E_centers_source1 = (Elo1 + Ehi1) / 2

        flux_arrays_source1 = {
            'numu': self.flux_data_source1['numu_flux'],
            'numub': self.flux_data_source1['numub_flux'],
            'nue': self.flux_data_source1['nue_flux'],
            'nueb': self.flux_data_source1['nueb_flux']
        }

        # 源2数据处理
        Elo2, Ehi2 = self.flux_data_source2['energy_bins_edges']
        self.E_centers_source2 = (Elo2 + Ehi2) / 2

        flux_arrays_source2 = {
            'numu': self.flux_data_source2['numu_flux'],
            'numub': self.flux_data_source2['numub_flux'],
            'nue': self.flux_data_source2['nue_flux'],
            'nueb': self.flux_data_source2['nueb_flux']
        }

        # 计算每个源的总通量
        self.total_flux_per_bin_source1 = sum(flux_arrays_source1.values())
        self.total_flux_per_bin_source2 = sum(flux_arrays_source2.values())

        self.total_flux_sum_source1 = np.sum(self.total_flux_per_bin_source1)
        self.total_flux_sum_source2 = np.sum(self.total_flux_per_bin_source2)

        self.total_four_flavors_source1 = sum(np.sum(f) for f in flux_arrays_source1.values())
        self.total_four_flavors_source2 = sum(np.sum(f) for f in flux_arrays_source2.values())

        # 计算归一化权重
        self.norm_weights_source1 = {}
        self.norm_weights_source2 = {}

        for flavor in ['numu', 'numub', 'nue', 'nueb']:
            self.norm_weights_source1[flavor] = (
                    flux_arrays_source1[flavor] / self.total_four_flavors_source1
            )
            self.norm_weights_source2[flavor] = (
                    flux_arrays_source2[flavor] / self.total_four_flavors_source2
            )

        # 计算合并权重（根据总通量比例）
        total_combined = self.total_four_flavors_source1 + self.total_four_flavors_source2
        self.source1_weight = self.total_four_flavors_source1 / total_combined
        self.source2_weight = self.total_four_flavors_source2 / total_combined

    def get_energy_spectrum(self, flavor='numub', source='combined'):
        """
        获取指定味和源的能量谱

        Parameters:
        -----------
        flavor : str
            中微子味 ('numu', 'numub', 'nue', 'nueb')
        source : str
            数据源 ('source1', 'source2', 'combined')

        Returns:
        --------
        E_centers : numpy array
            能量中心值数组
        weights : numpy array
            归一化权重数组
        """
        if source == 'source1':
            return self.E_centers_source1.copy(), self.norm_weights_source1[flavor].copy()
        elif source == 'source2':
            return self.E_centers_source2.copy(), self.norm_weights_source2[flavor].copy()
        elif source == 'combined':
            # 返回两个源的合并数据
            E_combined = np.concatenate([self.E_centers_source1, self.E_centers_source2])
            weights_combined = np.concatenate([
                self.norm_weights_source1[flavor] * self.source1_weight,
                self.norm_weights_source2[flavor] * self.source2_weight
            ])
            return E_combined, weights_combined
        else:
            raise ValueError(f"无效的源类型: {source}")


# ============================================================
# 2. 两味旋转矩阵（保持不变）
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
# 3. 构造不同模型的混合矩阵（保持不变）
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
                rotation_matrix(N, 2, 3, np.arcsin(0.15)) @
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
                rotation_matrix(N, 3, 4, np.arcsin(0.15)) @
                rotation_matrix(N, 2, 4, np.arcsin(0.15)) @
                rotation_matrix(N, 1, 4, np.arcsin(0.13)) @
                rotation_matrix(N, 0, 4, np.arcsin(0.14)) @
                rotation_matrix(N, 2, 3, np.arcsin(0.15)) @
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
                rotation_matrix(N, 3, 4, np.arcsin(0.15)) @
                rotation_matrix(N, 2, 4, np.arcsin(0.15)) @
                rotation_matrix(N, 1, 4, np.arcsin(0.17)) @
                rotation_matrix(N, 0, 4, np.arcsin(0.13)) @
                rotation_matrix(N, 2, 3, np.arcsin(0.15)) @
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
# 4. 能谱平均振荡概率（支持双源）
# ============================================================

class OscillationCalculator:
    """振荡概率计算器，支持双源"""

    def __init__(self, flux_data, model_builder):
        self.flux = flux_data
        self.models = model_builder
        # 两个源可以有不同基线
        self.source1_L_km = 0.5  # 源1的基线，单位km（可配置）
        self.source2_L_km = 0.3  # 源2的基线，单位km（可配置）

    def calculate_average_probability(self, model_name, initial_flavor, target_idx,
                                      L_km=None, source='combined'):
        """
        计算平均振荡概率，支持单源或双源合并

        Parameters:
        -----------
        model_name : str
            模型名称
        initial_flavor : str
            初始中微子味
        target_idx : int
            目标中微子索引
        L_km : float or None
            基线长度(km)。如果为None，则使用源特定的基线
        source : str
            数据源 ('source1', 'source2', 'combined')
        """
        U, m2, N = self.models.build_model(model_name)

        # 根据源类型计算概率
        if source == 'combined':
            # 分别计算两个源的权重
            E1, weights1 = self.flux.get_energy_spectrum(initial_flavor, 'source1')
            E2, weights2 = self.flux.get_energy_spectrum(initial_flavor, 'source2')

            # 分别计算两个源的基线
            L1 = L_km if L_km is not None else self.source1_L_km
            L2 = L_km if L_km is not None else self.source2_L_km

            # 计算两个源的概率
            prob1 = self._calculate_single_source_prob(
                U, m2, N, initial_flavor, target_idx, E1, weights1, L1
            )
            prob2 = self._calculate_single_source_prob(
                U, m2, N, initial_flavor, target_idx, E2, weights2, L2
            )

            # 根据总通量比例加权合并
            combined_prob = (
                                    prob1 * self.flux.total_four_flavors_source1 +
                                    prob2 * self.flux.total_four_flavors_source2
                            ) / (self.flux.total_four_flavors_source1 + self.flux.total_four_flavors_source2)

            return combined_prob

        else:
            # 单个源计算
            if source == 'source1':
                E_centers, flux_weights = self.flux.get_energy_spectrum(initial_flavor, 'source1')
                L = L_km if L_km is not None else self.source1_L_km
            else:  # source2
                E_centers, flux_weights = self.flux.get_energy_spectrum(initial_flavor, 'source2')
                L = L_km if L_km is not None else self.source2_L_km

            return self._calculate_single_source_prob(
                U, m2, N, initial_flavor, target_idx, E_centers, flux_weights, L
            )

    def _calculate_single_source_prob(self, U, m2, N, initial_flavor, target_idx,
                                      E_centers, flux_weights, L_km):
        """计算单个源的振荡概率"""
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

    def compute_all_probabilities(self, L_range=(0.001, 0.5), num_points=500, source='combined'):
        """
        计算所有模型和情况的振荡概率

        Parameters:
        -----------
        source : str
            数据源 ('source1', 'source2', 'combined')
        """
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
                        model_name, initial, target_idx, L, source
                    )
                model_probs[(initial, target_name)] = probs

            all_probs[model_name] = model_probs

        return L_m, all_probs, model_names


# ============================================================
# 5. 绘图函数（支持显示单源和合并结果）
# ============================================================

def plot_all_oscillation_probabilities_dual_source():
    """绘制双源振荡概率图，包含单源和合并结果"""
    print("双源中微子振荡概率计算")
    print("=" * 60)

    print("配置信息:")
    print("源1: 反中微子模式, 基线 = 0.5 km")
    print("源2: 中微子模式, 基线 = 0.3 km")
    print("=" * 60)

    # 初始化组件
    flux_data = FluxData()
    model_builder = NeutrinoModels()
    calculator = OscillationCalculator(flux_data, model_builder)

    # 计算三个版本的概率
    print("正在计算振荡概率...")

    # 1. 源1单独
    print("  计算源1单独...")
    L_m1, all_probs_source1, models1 = calculator.compute_all_probabilities(
        L_range=(0.001, 0.8),
        num_points=500,
        source='source1'
    )

    # 2. 源2单独
    print("  计算源2单独...")
    L_m2, all_probs_source2, models2 = calculator.compute_all_probabilities(
        L_range=(0.001, 0.8),
        num_points=500,
        source='source2'
    )

    # 3. 合并结果
    print("  计算合并结果...")
    L_m_combined, all_probs_combined, models_combined = calculator.compute_all_probabilities(
        L_range=(0.001, 0.8),
        num_points=500,
        source='combined'
    )

    print("计算完成！开始绘制图片...")

    # 计算各味总通量
    print(f"\n通量信息:")
    print(f"源1总通量: {flux_data.total_four_flavors_source1:.2e}")
    print(f"源2总通量: {flux_data.total_four_flavors_source2:.2e}")
    print(f"总合并通量: {flux_data.total_four_flavors_source1 + flux_data.total_four_flavors_source2:.2e}")
    print(f"源1权重: {flux_data.source1_weight:.3f}")
    print(f"源2权重: {flux_data.source2_weight:.3f}")

    # 创建2行3列的子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 定义六种情况
    cases = [
        {
            'row': 0, 'col': 0,
            'target': 'nue',
            'title': r'$\nu_e$ Appearance',
            'ylabel': r'$P(\nu_e)$ [%]'
        },
        {
            'row': 0, 'col': 1,
            'target': 'nueb',
            'title': r'$\bar{\nu}_e$ Appearance',
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
            'title': r'$\nu_\mu$ Survival',
            'ylabel': r'$P(\nu_\mu)$ [%]'
        },
        {
            'row': 1, 'col': 1,
            'target': 'numub',
            'title': r'$\bar{\nu}_\mu$ Survival',
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

        # 定义线型和颜色
        line_styles = {
            'source1': {'linestyle': '--', 'alpha': 0.7, 'label': 'Source 1'},
            'source2': {'linestyle': ':', 'alpha': 0.7, 'label': 'Source 2'},
            'combined': {'linestyle': '-', 'linewidth': 2, 'label': 'Combined'}
        }

        # 为每个模型绘制三种情况的线
        for model in models_combined:
            if case.get('type') == 'combined':
                # 组合情况
                if case['flavor'] == 'nue':
                    # 总电子中微子概率
                    total_probs_source1 = np.zeros_like(L_m1)
                    total_probs_source2 = np.zeros_like(L_m2)
                    total_probs_combined = np.zeros_like(L_m_combined)

                    # 源1
                    if ('numu', 'nue') in all_probs_source1[model]:
                        total_probs_source1 += all_probs_source1[model][('numu', 'nue')]
                    if ('nue', 'nue') in all_probs_source1[model]:
                        total_probs_source1 += all_probs_source1[model][('nue', 'nue')]

                    # 源2
                    if ('numu', 'nue') in all_probs_source2[model]:
                        total_probs_source2 += all_probs_source2[model][('numu', 'nue')]
                    if ('nue', 'nue') in all_probs_source2[model]:
                        total_probs_source2 += all_probs_source2[model][('nue', 'nue')]

                    # 合并
                    if ('numu', 'nue') in all_probs_combined[model]:
                        total_probs_combined += all_probs_combined[model][('numu', 'nue')]
                    if ('nue', 'nue') in all_probs_combined[model]:
                        total_probs_combined += all_probs_combined[model][('nue', 'nue')]

                else:  # νμ + 反νμ
                    # 总μ子中微子概率
                    total_probs_source1 = np.zeros_like(L_m1)
                    total_probs_source2 = np.zeros_like(L_m2)
                    total_probs_combined = np.zeros_like(L_m_combined)

                    # 源1
                    if ('numu', 'numu') in all_probs_source1[model]:
                        total_probs_source1 += all_probs_source1[model][('numu', 'numu')]

                    # 源2
                    if ('numu', 'numu') in all_probs_source2[model]:
                        total_probs_source2 += all_probs_source2[model][('numu', 'numu')]

                    # 合并
                    if ('numu', 'numu') in all_probs_combined[model]:
                        total_probs_combined += all_probs_combined[model][('numu', 'numu')]

                # 绘制三条线
                ax.plot(L_m1, total_probs_source1 * 100,
                        color=plt.cm.tab10(int(model) if model.isdigit() else 0),
                        **line_styles['source1'])
                ax.plot(L_m2, total_probs_source2 * 100,
                        color=plt.cm.tab10(int(model) if model.isdigit() else 0),
                        **line_styles['source2'])
                ax.plot(L_m_combined, total_probs_combined * 100,
                        color=plt.cm.tab10(int(model) if model.isdigit() else 0),
                        **line_styles['combined'], label=f'Model {model}')

            else:
                # 单个味的情况
                if case['target'] == 'nue':
                    # νe概率
                    probs_source1 = np.zeros_like(L_m1)
                    probs_source2 = np.zeros_like(L_m2)
                    probs_combined = np.zeros_like(L_m_combined)

                    # 源1
                    if ('numu', 'nue') in all_probs_source1[model]:
                        probs_source1 += all_probs_source1[model][('numu', 'nue')]
                    if ('nue', 'nue') in all_probs_source1[model]:
                        probs_source1 += all_probs_source1[model][('nue', 'nue')]

                    # 源2
                    if ('numu', 'nue') in all_probs_source2[model]:
                        probs_source2 += all_probs_source2[model][('numu', 'nue')]
                    if ('nue', 'nue') in all_probs_source2[model]:
                        probs_source2 += all_probs_source2[model][('nue', 'nue')]

                    # 合并
                    if ('numu', 'nue') in all_probs_combined[model]:
                        probs_combined += all_probs_combined[model][('numu', 'nue')]
                    if ('nue', 'nue') in all_probs_combined[model]:
                        probs_combined += all_probs_combined[model][('nue', 'nue')]

                elif case['target'] == 'nueb':
                    # 反νe概率
                    probs_source1 = np.zeros_like(L_m1)
                    probs_source2 = np.zeros_like(L_m2)
                    probs_combined = np.zeros_like(L_m_combined)

                    # 源1
                    if ('numub', 'nueb') in all_probs_source1[model]:
                        probs_source1 += all_probs_source1[model][('numub', 'nueb')]
                    if ('nueb', 'nueb') in all_probs_source1[model]:
                        probs_source1 += all_probs_source1[model][('nueb', 'nueb')]

                    # 源2
                    if ('numub', 'nueb') in all_probs_source2[model]:
                        probs_source2 += all_probs_source2[model][('numub', 'nueb')]
                    if ('nueb', 'nueb') in all_probs_source2[model]:
                        probs_source2 += all_probs_source2[model][('nueb', 'nueb')]

                    # 合并
                    if ('numub', 'nueb') in all_probs_combined[model]:
                        probs_combined += all_probs_combined[model][('numub', 'nueb')]
                    if ('nueb', 'nueb') in all_probs_combined[model]:
                        probs_combined += all_probs_combined[model][('nueb', 'nueb')]

                elif case['target'] == 'numu':
                    # νμ概率
                    probs_source1 = all_probs_source1[model][('numu', 'numu')]
                    probs_source2 = all_probs_source2[model][('numu', 'numu')]
                    probs_combined = all_probs_combined[model][('numu', 'numu')]

                elif case['target'] == 'numub':
                    # 反νμ概率
                    probs_source1 = all_probs_source1[model][('numub', 'numub')]
                    probs_source2 = all_probs_source2[model][('numub', 'numub')]
                    probs_combined = all_probs_combined[model][('numub', 'numub')]

                # 绘制三条线
                ax.plot(L_m1, probs_source1 * 100,
                        color=plt.cm.tab10(int(model) if model.isdigit() else 0),
                        **line_styles['source1'])
                ax.plot(L_m2, probs_source2 * 100,
                        color=plt.cm.tab10(int(model) if model.isdigit() else 0),
                        **line_styles['source2'])
                ax.plot(L_m_combined, probs_combined * 100,
                        color=plt.cm.tab10(int(model) if model.isdigit() else 0),
                        **line_styles['combined'], label=f'Model {model}')

        ax.set_xlabel("Baseline L (m)", fontsize=11)
        ax.set_ylabel(case['ylabel'], fontsize=11)
        ax.set_title(case['title'], fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(L_m_combined))

    # 在第一个子图显示图例
    axes[0, 0].legend(fontsize=9, loc='upper left')

    # 添加源信息的文本框
    info_text = f"Source 1: Anti-neutrino mode, L = {calculator.source1_L_km:.1f} km\n" \
                f"Source 2: Neutrino mode, L = {calculator.source2_L_km:.1f} km\n" \
                f"Weights: Source1={flux_data.source1_weight:.2f}, " \
                f"Source2={flux_data.source2_weight:.2f}"

    fig.text(0.02, 0.98, info_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # 调整布局
    plt.suptitle(
        r"Dual-Source Neutrino Oscillation Probabilities for Different Models",
        fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图片
    plt.savefig('dual_source_oscillation_probabilities.png', dpi=150, bbox_inches='tight')
    print("已保存图片: dual_source_oscillation_probabilities.png")

    # 显示图片
    plt.show()


# ============================================================
# 运行主程序
# ============================================================

if __name__ == "__main__":
    # 可以配置不同的源参数
    plot_all_oscillation_probabilities_dual_source()