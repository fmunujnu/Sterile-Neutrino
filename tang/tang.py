import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_fwhm_approx(lambda_c, A, c=1.5, L=3.0, lambda_0=153.0):
    """
    使用一阶微分近似公式计算 FWHM (Δλ)
    Δλ ≈ [π * (λ_c^2 - λ_0^2)^2] / [4 * A * c * L * λ_c]
    
    注意：此处为了让角度单位匹配，公式中的 π (弧度) 在对应度制下的 A 时，
    应转换为度制 (π rad = 180 deg) 进行量级对齐。
    """
    k = A * c * L
    numerator = 180.0 * (lambda_c**2 - lambda_0**2)**2
    denominator = 4.0 * k * lambda_c
    return numerator / denominator

def main():
    # 1. 设定物理范围 (自适应调整)
    # 中心波长 lambda_c 范围: 380 nm 至 780 nm (可见光波段)
    lambda_c_axis = np.linspace(400, 750, 200)
    
    # Drude 常数 A 范围: 1.5e7 到 3.0e7
    A_scale_axis = np.linspace(1.5, 3.0, 200)  # 单位: 10^7
    
    # 2. 生成网格
    Lambda_C, A_scale = np.meshgrid(lambda_c_axis, A_scale_axis)
    A_actual = A_scale * 1e7
    
    # 3. 计算因变量 FWHM (Δλ)
    FWHM = calculate_fwhm_approx(Lambda_C, A_actual, c=1.5, L=3.0, lambda_0=153.0)
    
    # 4. 开始绘制 3D 着色曲面与投影等高线
    fig = plt.figure(figsize=(13, 9), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    
    # 自适应Z轴显示范围：让动态变化充满整个视觉空间
    z_min, z_max = np.min(FWHM), np.max(FWHM)
    
    # 绘制三维曲面
    surf = ax.plot_surface(Lambda_C, A_scale, FWHM, 
                           cmap=cm.plasma, 
                           linewidth=0, 
                           antialiased=True, 
                           alpha=0.85, 
                           rstride=2, cstride=2)
    
    # 5. 绘制多平面立体投影 (等高线图)
    # Z轴底部投影 (offset 设为 z_min)
    ax.contourf(Lambda_C, A_scale, FWHM, 
                zdir='z', offset=z_min, 
                cmap=cm.plasma, alpha=0.3)
    ax.contour(Lambda_C, A_scale, FWHM, 
               zdir='z', offset=z_min, 
               colors='black', linewidths=0.6, alpha=0.5)
    
    # X轴侧面投影
    ax.contour(Lambda_C, A_scale, FWHM, 
               zdir='x', offset=400, 
               cmap=cm.plasma, alpha=0.3)
    
    # Y轴侧面投影
    ax.contour(Lambda_C, A_scale, FWHM, 
               zdir='y', offset=1.5, 
               cmap=cm.plasma, alpha=0.3)

    # 6. 设置轴标签与精细外观
    ax.set_xlabel('中心波长 $\lambda_c$ (nm)', fontsize=11, labelpad=10)
    ax.set_ylabel('色散系数 $A$ ($10^7$ $\\rm{deg\\cdot nm^2\\cdot mL/(dm\\cdot g)}$)', fontsize=11, labelpad=10)
    ax.set_zlabel('半高全宽 $\\Delta\\lambda$ (nm)', fontsize=11, labelpad=10)
    
    ax.set_title('单色仪带宽 $\\Delta\\lambda$ 随参数 $\\lambda_c, A$ 的变化', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 精准控制视角，消除视觉压平感
    ax.set_xlim(400, 750)
    ax.set_ylim(1.5, 3.0)
    ax.set_zlim(z_min, z_max)
    
    # 调整视角 (提升俯角 elev 和方位角 azim，强化曲面的陡峭感)
    ax.view_init(elev=30, azim=-50)
    
    # 添加色带
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.08)
    
    plt.tight_layout()
    plt.savefig('fwhm_3d_simulation_fixed.png', dpi=300, bbox_inches='tight')
    print("修正后的图像已成功保存为: fwhm_3d_simulation_fixed.png")
    plt.show()

if __name__ == '__main__':
    main()