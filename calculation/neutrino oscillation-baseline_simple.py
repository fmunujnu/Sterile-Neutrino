import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 能谱与初始味态（统一定义）
# ============================================================

E_bins = np.array([1])      # GeV
flux = np.array([1])

# 保存原始数组用于比较
original_flux = flux.copy()
original_sum = np.sum(original_flux)

# 直接修改原数组（归一化到总和为1）
flux[:] = flux / np.sum(flux)

# 初始味态 (νe, νμ, ντ, νs1, νs2)
initial_flavor = np.array([0.04, 0.96, 0.0, 0.0, 0.0])

# ============================================================
# 2. 两味旋转矩阵
# ============================================================

def rotation_matrix(N, i, j, theta, delta=0.0):
    R = np.identity(N, dtype=complex)
    R[i, i] = np.cos(theta)
    R[j, j] = np.cos(theta)
    R[i, j] = np.sin(theta) * np.exp(-1j * delta)
    R[j, i] = -np.sin(theta) * np.exp(1j * delta)
    return R

# ============================================================
# 3. 构造不同模型的混合矩阵
# ============================================================

def build_model(model):
    if model == "3":
        N = 3
        U = (
                rotation_matrix(N, 1, 2, 49 * np.pi / 180) @
                rotation_matrix(N, 0, 2, 8.6 * np.pi / 180) @
                rotation_matrix(N, 0, 1, 33 * np.pi / 180)
        )
        m2 = np.array([0.0, 7.5e-5, 2.5e-3])

    elif model == "3+1":
        N = 4
        U = (
            rotation_matrix(N, 2, 3, np.arcsin(0.15)) @#
            rotation_matrix(N, 1, 3, np.arcsin(0.17)) @
            rotation_matrix(N, 0, 3, np.arcsin(0.15)) @
            rotation_matrix(N, 1, 2, 41 * np.pi / 180) @
            rotation_matrix(N, 0, 2, 8.44 * np.pi / 180) @
            rotation_matrix(N, 0, 1, 34.5 * np.pi / 180)
        )
        m2 = np.array([0.0, 7.5e-5, 2.5e-3, 1])

    elif model == "3+2":
        N = 5
        U = (
            rotation_matrix(N, 3, 4, np.arcsin(0.15)) @#
            rotation_matrix(N, 2, 4, np.arcsin(0.15)) @#
            rotation_matrix(N, 1, 4, np.arcsin(0.13)) @
            rotation_matrix(N, 0, 4, np.arcsin(0.14)) @
            rotation_matrix(N, 2, 3, np.arcsin(0.15)) @#
            rotation_matrix(N, 1, 3, np.arcsin(0.15)) @
            rotation_matrix(N, 0, 3, np.arcsin(0.13)) @
            rotation_matrix(N, 1, 2, 41 * np.pi / 180) @
            rotation_matrix(N, 0, 2, 8.44 * np.pi / 180) @
            rotation_matrix(N, 0, 1, 34.5 * np.pi / 180)
        )
        m2 = np.array([0.0, 7.5e-5, 2.5e-3, 0.47, 0.87])

    elif model == "1+3+1":
        N = 5
        U = (
            rotation_matrix(N, 3, 4, np.arcsin(0.15)) @#
            rotation_matrix(N, 2, 4, np.arcsin(0.15)) @#
            rotation_matrix(N, 1, 4, np.arcsin(0.17)) @
            rotation_matrix(N, 0, 4, np.arcsin(0.13)) @
            rotation_matrix(N, 2, 3, np.arcsin(0.15)) @#
            rotation_matrix(N, 1, 3, np.arcsin(0.13)) @
            rotation_matrix(N, 0, 3, np.arcsin(0.15)) @
            rotation_matrix(N, 1, 2, 41*np.pi/180) @
            rotation_matrix(N, 0, 2, 8.44*np.pi/180) @
            rotation_matrix(N, 0, 1, 34.5*np.pi/180)
        )
        m2 = np.array([0.0, 7.5e-5, 2.5e-3, -0.87, 0.47])


    else:
        raise ValueError("Unknown model")

    return U, m2, N

# ============================================================
# 4. 能谱平均振荡概率
# ============================================================

def averaged_probability(U, m2, alpha_dist, beta, L_km):
    prob = 0.0
    for E, w in zip(E_bins, flux):
        phase = np.exp(-1j * 1.267 * m2 * L_km / E)
        amp = 0.0 + 0.0j
        for a in range(len(alpha_dist)):
            amp += alpha_dist[a] * np.sum(
                U[beta, :] * np.conj(U[a, :]) * phase
            )
        prob += w * np.abs(amp)**2
    return prob

# ============================================================
# 5. 分图绘制函数（νe / νμ）
# ============================================================

def plot_single_flavor(L_km, beta, flavor_label):
    models = ["3", "3+1", "3+2", "1+3+1"]

    L_m = L_km * 1000.0
    plt.figure(figsize=(10, 7))

    for model in models:
        U, m2, N = build_model(model)
        P = [
            averaged_probability(U, m2, initial_flavor[:N], beta, L)
            for L in L_km
        ]
        plt.plot(L_m, np.array(P) * 100, linewidth=2, label=model)

    plt.xlabel("Baseline L (m)", fontsize=13)
    plt.ylabel(rf"$P({flavor_label})$ [%]", fontsize=13)
    plt.title(rf"Neutrino Oscillation: ${flavor_label}$", fontsize=15)

    # 自动 y 轴缩放（不要再设置 ylim）
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

# ============================================================
# 6. 主程序
# ============================================================

# 1–1000 m（线性）
L_km = np.linspace(0.001, 0.06, 1000)

# νe 生存概率
plot_single_flavor(L_km, beta=0, flavor_label="νe")

# νμ 出现概率
plot_single_flavor(L_km, beta=1, flavor_label="νμ")
