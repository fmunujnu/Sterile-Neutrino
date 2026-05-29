import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_fwhm(lambda_c, A, c=1.5, L=3.0, lambda_0=153.0):
    """
    基于严格解析式计算 FWHM (\Delta\lambda)
    
    参数:
    lambda_c : 系统的中心透射波长 (nm)
    A        : Drude 色散常数 (deg * nm^2 * mL / (dm * g))
               在实验中, 典型值为 A = 2.165e7 deg*nm^2*mL/(dm*g) (若波长单位为nm)
               为了画图美观，我们将A的量级做归一化，或直接按物理尺度计算。
    c        : 溶液浓度 (g/mL), 默认 1.5 g/mL (接近饱和)
    L        : 样品管光程 (dm), 默认 3.0 dm (30 cm)
    lambda_0 : 蔗糖共振截止波长 (nm), 默认 153.0 nm
    """
    # 计算旋光色散系数 k = A * c * L
    k = A * c * L
    
    # 计算中间项
    term_c = k / (lambda_c**2 - lambda_0**2)
    
    # 物理合理性条件判定：分母项必须大于 pi/4 (即 0.7854), 否则该波长下包络断裂，无法形成半高宽
    # 我们使用 mask 来滤去不符合物理实际的奇点区域
    denom_minus = term_c - np.pi / 4
    denom_plus = term_c + np.pi / 4
    
    # 过滤无效区域
    valid_mask = (denom_minus > 0) & (lambda_c > lambda_0)
    
    # 初始化输出数组
    fwhm = np.full_like(lambda_c, np.nan)
    
    # 针对有效物理区域计算严格半高宽
    term1 = lambda_0**2 + k / denom_minus
    term2 = lambda_0**2 + k / denom_plus
    
    # 确保根号下非负
    valid_mask = valid_mask & (term1 >= 0) & (term2 >= 0)
    
    fwhm[valid_mask] = np.sqrt(term1[valid_mask]) - np.sqrt(term2[valid_mask])
    return fwhm

def main():
    # 1. 设定物理范围
    # 中心波长 lambda_c 范围: 380 nm 至 780 nm (可见光波段)
    lambda_c_axis = np.linspace(380, 780, 200)
    
    # Drude 常数 A 范围: 典型值约 2.165e7，我们取其在 1.0e7 到 3.5e7 之间波动以观察参数敏感性
    # 为了坐标轴刻度美观，我们将 A 缩放到 10^7 量级
    A_scale_axis = np.linspace(1.0, 3.5, 200)  # 单位: 10^7 deg*nm^2*mL/(dm*g)
    
    # 2. 生成网格
    Lambda_C, A_scale = np.meshgrid(lambda_c_axis, A_scale_axis)
    A_actual = A_scale * 1e7
    
    # 3. 计算因变量 FWHM (\Delta\lambda)
    FWHM = calculate_fwhm(Lambda_C, A_actual, c=1.5, L=3.0, lambda_0=153.0)
    
    # 对含有NaN的矩阵进行插值或边界处理，便于画出平滑的等高线
    # 在这里，因为物理截止是自然的，保留 NaN 会让 Matplotlib 自动切除不合理的物理区域
    
    # 4. 开始绘制 3D 着色曲面与投影等高线
    fig = plt.figure(figsize=(14, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    
    # 自适应Z轴范围
    max_fwhm_limit = 150.0  # 限制最大显示半高宽为 150nm，避免奇点导致数值发散影响整体视觉
    FWHM_clipped = np.clip(FWHM, 0, max_fwhm_limit)
    
    # 绘制三维曲面 (使用 colormap 'viridis' 或 'plasma' 营造高级学术感)
    surf = ax.plot_surface(Lambda_C, A_scale, FWHM_clipped, 
                           cmap=cm.plasma, 
                           linewidth=0, 
                           antialiased=True, 
                           alpha=0.85, 
                           rstride=2, cstride=2,
                           label='FWHM 响应曲面')
    
    # 5. 绘制立体投影 (等高线图)
    # Z轴投影 (投影到底部，设定 offset 为 Z 轴的下限)
    z_offset = 0
    ax.contourf(Lambda_C, A_scale, FWHM_clipped, 
                zdir='z', offset=z_offset, 
                cmap=cm.plasma, alpha=0.3)
    
    ax.contour(Lambda_C, A_scale, FWHM_clipped, 
               zdir='z', offset=z_offset, 
               colors='black', linewidths=0.8, alpha=0.5)
    
    # X轴投影 (投影到侧面 $\lambda_c$ 下限)
    ax.contour(Lambda_C, A_scale, FWHM_clipped, 
               zdir='x', offset=380, 
               cmap=cm.plasma, alpha=0.4)
    
    # Y轴投影 (投影到侧面 A 下限)
    ax.contour(Lambda_C, A_scale, FWHM_clipped, 
               zdir='y', offset=1.0, 
               cmap=cm.plasma, alpha=0.4)

    # 6. 设置轴标签与精细外观
    ax.set_xlabel('中心波长 $\lambda_c$ (nm)', fontsize=12, labelpad=10)
    ax.set_ylabel('色散系数 $A$ ($10^7$ $\\rm{deg\\cdot nm^2\\cdot mL/(dm\\cdot g)}$)', fontsize=12, labelpad=10)
    ax.set_zlabel('半高全宽 $\\Delta\\lambda$ (nm)', fontsize=12, labelpad=10)
    
    ax.set_title('单级单色仪带宽 $\\Delta\\lambda$ 在参数空间 $(\\lambda_c, A)$ 的理论预测与立体投影', 
                 fontsize=15, fontweight='bold', pad=20)
    
    # 设置刻度范围
    ax.set_xlim(380, 780)
    ax.set_ylim(1.0, 3.5)
    ax.set_zlim(z_offset, max_fwhm_limit)
    
    # 添加高级学术色带
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label('带宽 FWHM (nm)', fontsize=11)
    
    # 优化视角：便于评委同时看清曲面的弯曲走势与底部等高线
    ax.view_init(elev=28, azim=-125)
    
    # 7. 物理边界标注和解释说明 (作为图上嵌入文本，提升答辩专业度)
    info_text = (
        "物理机制判据：\n"
        "1. 当 A 增大且 λ_c 减小时, FWHM 急剧变窄(蓝紫光区分辨率本征优于红光区)\n"
        "2. 长波奇点风险：当 A*c*L/(λ_c^2 - λ_0^2) < π/4 时，透射包络断裂\n"
        "3. 实验设计偏导数：∂(Δλ)/∂c < 0, ∂(Δλ)/∂L < 0"
    )
    fig.text(0.12, 0.05, info_text, fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    
    # 保存高质量图片，便于直接拖入 Beamer 幻灯片
    plt.savefig('fwhm_3d_simulation.png', dpi=300, bbox_inches='tight')
    print("图像已成功绘制并保存为高质量图片: fwhm_3d_simulation.png")
    
    plt.show()

if __name__ == '__main__':
    main()