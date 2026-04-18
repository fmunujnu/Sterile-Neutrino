import numpy as np
import matplotlib.pyplot as plt

def plot_bkg_pred_vs_data(background=None, signal=None, observed=None, bin_edges=None,
                          model_name=None, param_label=None,
                          savepath=None, title=None, xlabel=None, ylabel=None):
    """
    绘制“本底 + 模型预测信号”与观测数据的对比折线图。
    如果未提供数据，则使用内部硬编码的示例数据（MiniBooNE 样例）。

    Parameters
    ----------
    background : array-like or None, shape (n_bins,)
        本底计数谱。若为 None，则使用硬编码数据。
    signal : array-like or None, shape (n_bins,)
        信号计数谱。若为 None，则使用硬编码数据。
    observed : array-like or None, shape (n_bins,)
        观测数据计数谱。若为 None，则使用硬编码数据。
    bin_edges : array-like or None, shape (n_bins+1,)
        能量 bin 边界（单位：GeV）。若为 None，则使用硬编码数据。
    model_name : str, optional
        模型名称，用于图标题。
    param_label : str or None, optional
        参数描述文字。
    savepath : str or None, optional
        若不为 None，则将图保存到该路径。
    title : str or None, optional
        覆盖默认标题。
    xlabel : str or None, optional
        x 轴标签。
    ylabel : str or None, optional
        y 轴标签。

    Returns
    -------
    fig, ax : matplotlib Figure 与 Axes 对象。
    """
    # 如果未提供任何数据，使用内部硬编码数据
    if background is None or signal is None or observed is None or bin_edges is None:
        # 硬编码示例数据（来自 MiniBooNE 样例）
        total_pred = np.array([10.289156626506024, 13.240963855421683, 14.16867469879518, 9.783132530120481, 2.1927710843373482])
        background = np.array([4.8915662650602405, 5.144578313253012, 3.626506024096385, 2.1927710843373482, 0.59036144578313])
        observed = np.array([10.204819277108435, 11.638554216867465, 17.542168674698793, 7.759036144578315, 3.7108433734939754])
        bin_edges = np.array([20, 28, 36, 44, 52, 60])
        signal = total_pred - background   # 信号 = 总预测 - 本底

    # 转换为数组
    background = np.asarray(background)
    signal = np.asarray(signal)
    observed = np.asarray(observed)
    bin_edges = np.asarray(bin_edges)

    # 检查维度
    n_bins = len(background)
    if not (len(signal) == n_bins and len(observed) == n_bins and len(bin_edges) == n_bins + 1):
        raise ValueError("数组长度不一致：background, signal, observed 应有相同长度，bin_edges 长度应多1")

    # 计算总预测谱、能量中心、x误差
    total_pred = background + signal
    energies = (bin_edges[:-1] + bin_edges[1:]) / 2.0          # bin 中心
    xerr_lo = energies - bin_edges[:-1]                        # 左侧半宽
    xerr_hi = bin_edges[1:] - energies                         # 右侧半宽

    # 统计误差（泊松近似）
    data_err = np.sqrt(np.maximum(observed, 1.0))
    pred_err = np.sqrt(np.maximum(total_pred, 1.0))

    # 准备步进图数据：x 使用 bin_edges（长度 n_bins+1），y 扩展最后一个值使长度一致
    step_x = bin_edges
    step_bkg = np.append(background, background[-1])          # 长度 n_bins+1
    step_pred = np.append(total_pred, total_pred[-1])         # 长度 n_bins+1
    step_pred_err = np.append(pred_err, pred_err[-1])         # 长度 n_bins+1

    # 开始绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 本底（虚线）
    ax.step(step_x, step_bkg, where='post',
            color='steelblue', linestyle='--', linewidth=1.6,
            label='Background only')

    # 本底+信号（实线 + 误差阴影）
    ax.step(step_x, step_pred, where='post',
            color='steelblue', linestyle='-', linewidth=2.0,
            label='Bkg + prediction (signal)')
    ax.fill_between(step_x, step_pred - step_pred_err, step_pred + step_pred_err,
                    step='post', color='steelblue', alpha=0.18,
                    label=r'Prediction $\pm\sqrt{N}$')

    # 观测数据（点 + 误差棒）
    ax.errorbar(energies, observed,
                yerr=data_err,
                xerr=[xerr_lo, xerr_hi],
                fmt='o', color='crimson', markersize=7,
                capsize=4, elinewidth=1.5, markeredgewidth=1.2,
                label='Observed data')

    # 文本框：总计数信息
    total_bkg = np.sum(background)
    total_signal = np.sum(signal)
    total_pred_sum = total_bkg + total_signal
    total_data = np.sum(observed)
    ax.text(0.97, 0.97,
            (f"Total bkg   = {total_bkg:.1f}\n"
             f"Total signal = {total_signal:.2f}\n"
             f"Total pred   = {total_pred_sum:.1f}\n"
             f"Total data   = {total_data:.0f}"),
            transform=ax.transAxes, fontsize=11,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.85))

    # 轴标签
    ax.set_xlabel(xlabel if xlabel is not None else r'Reconstructed $E_\nu^{\rm QE}$ [GeV]', fontsize=13)
    ax.set_ylabel(ylabel if ylabel is not None else 'Event count', fontsize=13)

    # 标题
    if title is not None:
        ax.set_title(title, fontsize=12)
    else:
        title_str = ''
        if model_name:
            title_str += f'LSND'
        title_str += 'Bkg + Prediction vs Data'
        if param_label:
            title_str += '\n' + r'$\bf{Params:}$ ' + param_label
        ax.set_title(title_str, fontsize=12)

    ax.legend(fontsize=11, loc='lower left', framealpha=0.85)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_xlim(bin_edges[0] * 0.98, bin_edges[-1] * 1.02)
    ax.set_ylim(bottom=0.)

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"  [plot_bkg_pred_vs_data] 图像已保存至: {savepath}")

    plt.show()
    return fig, ax

if __name__ == "__main__":
    # 直接运行脚本时，使用硬编码数据生成示例图
    plot_bkg_pred_vs_data(model_name="Example (Hardcoded Data)")