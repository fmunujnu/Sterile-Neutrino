# 现有实验的输入、重加权、统计方法与统一模型接口

> [!important] 文档范围
> 本文描述的是当前仓库活动代码实际执行的计算，而不是对官方分析流程的理想化转述。
> “测试通过”只证明程序满足仓库中已经声明的数值约定，不等于完整复现合作组内部分析。
> 当前三个实验的数据公开程度不同，因此共享振荡概率和输出接口，但不能强迫它们使用同一种实验 likelihood。

## 1. 总体计算框架

### 1.1 符号和事件预测

对一个有分箱数据的实验，令：

- $r$：重建量的 bin 编号；当前主要是重建中微子能量 bin；
- $t$：真实中微子能量 bin 编号；
- $E_t$：第 $t$ 个真实能量 bin 的代表能量；
- $L_t$：对应事件或真值 bin 的传播距离；若实验只提供单一基线，则所有 $t$ 使用同一个 $L$；
- $\alpha$：束流产生时的中微子味，例如 $e$ 或 $\mu$；
- $\beta$：被分析选择接受的末态味道，例如电子味或缪子味；
- $\boldsymbol\eta$：振荡模型参数的集合；
- $P_{\alpha\rightarrow\beta}(E_t,L_t;\boldsymbol\eta)$：在参数 $\boldsymbol\eta$ 下，中微子从味道 $\alpha$ 转变为味道 $\beta$ 的概率；
- $D_r$：实验在第 $r$ 个重建 bin 中观测到的事件数；
- $M_r(\boldsymbol\eta)$：模型在该 bin 中预测的总事件数；
- $B_r^{\rm fixed}$：当前适配器中不随振荡参数变化的固定背景。

如果拥有完整探测器模拟，预测可以写成

$$
M_r(\boldsymbol\eta)
=B_r^{\rm fixed}
+\sum_{\alpha,\beta,t}
\Phi_{\alpha t}\,
\sigma_{\beta t}\,
\epsilon_{\beta t}\,
R_{rt}^{\alpha\rightarrow\beta}\,
P_{\alpha\rightarrow\beta}(E_t,L_t;\boldsymbol\eta).
$$

这里：

- $\Phi_{\alpha t}$ 是初始味道 $\alpha$ 在真实能量 bin $t$ 中的通量；
- $\sigma_{\beta t}$ 是产生末态味道 $\beta$ 所对应可见反应的截面；
- $\epsilon_{\beta t}$ 是触发、重建和选择效率；
- $R_{rt}^{\alpha\rightarrow\beta}$ 是真实能量 $t$ 向重建 bin $r$ 的迁移概率或响应；
- 暴露量、靶核数、几何接受度等归一化因子也应包含在事件权重中。

仓库没有为每个实验分别得到上述所有数组，因此把振荡概率以外的已知或有效因素合并为过程核：

$$
K_{rt}^{\alpha\rightarrow\beta}
\equiv
\Phi_{\alpha t}\,
\sigma_{\beta t}\,
\epsilon_{\beta t}\,
R_{rt}^{\alpha\rightarrow\beta}
\times(\text{暴露量与归一化}).
$$

实际预测接口因此是

$$
\boxed{
M_r(\boldsymbol\eta)
=B_r^{\rm fixed}
+\sum_{\alpha,\beta,t}
K_{rt}^{\alpha\rightarrow\beta}
P_{\alpha\rightarrow\beta}(E_t,L_t;\boldsymbol\eta)
}.
$$

> [!warning] 防止重复加权
> 当前保存的 `*_response_counts.csv` 已经包含有效通量、响应以及由参考谱反推的逐重建-bin比例。
> 在扫描时只再乘振荡概率，不能重新乘一次 flux 或 Reco 矩阵。

### 1.2 统一接口和实验适配器

活动代码采用以下职责划分：

```text
实验公开文件
    ↓
experiments/<experiment>/adapter.py
    ↓
统一的预测或 -2 ln L 接口
    ↓
core/：3+1、1+3+1、profile、协方差和 CLs
    ↓
output.py：CSV、JSON、谱图和参数空间图
```

公共接口声明位于：

```text
src/sterile_fit/experiments/interface.py
```

所有实验至少应能提供某个参数点的

$$
-2\ln L_i(\boldsymbol\eta),
$$

其中下标 $i$ 表示第 $i$ 个实验。若不同实验统计独立，全局目标量可以相加：

$$
-2\ln L_{\rm global}(\boldsymbol\eta)
=\sum_i[-2\ln L_i(\boldsymbol\eta)].
$$

如果两个样本存在交叉协方差，例如同一 MicroBooNE 探测器中的 BNB 与 NuMI 样本，则不能当成两个独立项直接相加；必须先在实验适配器内组成一个联合向量和联合协方差。

### 1.3 当前实验支持等级

| 实验项 | 当前输入等级 | 当前用途 | 当前是否进入统一 $1+3+1$ 扫描 |
|---|---|---|---|
| MicroBooNE BNB 四通道 | 经验 kernel + 公开谱 + 协方差 | BNB-only 或联合扫描 | 是 |
| MicroBooNE BNB+NuMI 八通道 | 经验 kernel + 公开联合协方差 | 当前 Fig. 3a/3b 主要扫描 | 是，但 NuMI 探测器响应近似明显 |
| MiniBooNE $\nu+\bar\nu$ appearance | 公开逐事件 full-transmutation 样本 + 谱 + 协方差 | 独立 $3+1$ 二维验证 | 否；当前只实现 appearance-only $3+1$ |
| LSND DAR | 公开总转换概率 + 解析束流/截面近似 | 独立 $3+1$ 总率 likelihood | 否；接口可替换概率，但尚无正式 $1+3+1$ 扫描 |

---

## 2. 共同的振荡模型

### 2.1 当前 $3+1$ 参数

$3+1$ 模型在三个标准质量态之外增加一个质量本征态 $\nu_4$。当前基础参数为

$$
\boldsymbol\eta_{3+1}
=\left(
\Delta m_{41}^2,
s_{14},
s_{24}
\right),
$$

其中：

- $\Delta m_{41}^2=m_4^2-m_1^2$ 是额外态与第一质量态之间的质量平方差，单位为 $\mathrm{eV}^2$；
- $s_{14}\equiv\sin^2\theta_{14}$；
- $s_{24}\equiv\sin^2\theta_{24}$；
- $\theta_{14}$ 和 $\theta_{24}$ 是当前旋转约定中的混合角。

在代码采用的约定下，混合矩阵元素满足

$$
|U_{e4}|^2=s_{14},
\qquad
|U_{\mu4}|^2=(1-s_{14})s_{24}.
$$

$U_{\alpha i}$ 表示味道态 $\nu_\alpha$ 中质量态 $\nu_i$ 的复振幅。

短基线振荡相位为

$$
\Delta_{41}
=1.267\,
\frac{\Delta m_{41}^2[\mathrm{eV}^2]L[\mathrm{km}]}
{E_\nu[\mathrm{GeV}]},
$$

其中 $E_\nu$ 是中微子能量，$L$ 是传播距离。

程序统一计算四类电子/缪子味概率：

$$
P_{ee},\qquad P_{\mu e},\qquad P_{e\mu},\qquad P_{\mu\mu},
$$

并可分别处理对应的反中微子概率。在真空 $3+1$、且没有可见 CP 干涉的情况下，中微子和反中微子的这些概率相同；保留反中微子接口是为了以后接入 $1+3+1$ 的相位差异。

两个常用有效振幅定义为

$$
\sin^2(2\theta_{\mu e})
\equiv4|U_{e4}|^2|U_{\mu4}|^2
=4s_{14}(1-s_{14})s_{24},
$$

$$
\sin^2(2\theta_{ee})
\equiv4|U_{e4}|^2(1-|U_{e4}|^2)
=4s_{14}(1-s_{14}).
$$

它们分别控制 $\nu_\mu\rightarrow\nu_e$ appearance 和 $\nu_e\rightarrow\nu_e$ disappearance：

$$
P_{\mu e}
=\sin^2(2\theta_{\mu e})\sin^2\Delta_{41},
$$

$$
P_{ee}
=1-\sin^2(2\theta_{ee})\sin^2\Delta_{41}.
$$

### 2.2 未来统一的 $1+3+1$ 参数

$1+3+1$ 模型在三个近似退化的标准质量态两侧各放置一个额外态。当前代码使用

$$
\boldsymbol\eta_{1+3+1}
=\left(
|\Delta m_{41}^2|,
\Delta m_{51}^2,
|U_{e4}|^2,
|U_{\mu4}|^2,
|U_{e5}|^2,
|U_{\mu5}|^2,
\phi_{\mu e}
\right).
$$

这里：

- 代码规定 $\Delta m_{41}^2=-|\Delta m_{41}^2|<0$，即状态4位于三个轻态下方；
- $\Delta m_{51}^2>0$，即状态5位于三个轻态上方；
- 四个 $|U_{\alpha i}|^2$ 是混合矩阵元素的模平方，不是四个欧拉旋转角；
- $\phi_{\mu e}$ 是电子—缪子 appearance 中可测的相对 CP 相位，范围为 $[-\pi,\pi]$；
- 反中微子使用相反的 CP 相位符号。

当前代码还检查电子行、缪子行归一化，以及这两个重态片段能否嵌入一个幺正 $5\times5$ 混合矩阵。当前 CC 模型没有建立 $\tau$ 味和 NC 的完整响应，所以没有引入 $\theta_{34}$、$\theta_{35}$ 等额外参数。

对实验适配器而言，$3+1$ 与 $1+3+1$ 的理想差别只应是传入的概率函数不同：

```text
同一实验 kernel + P^(3+1)(E,L)
同一实验 kernel + P^(1+3+1)(E,L)
```

但这种替换只有在 kernel 真正按初始味道、末态味道、真能量和必要的基线信息分解时才可靠。只提供二维 $3+1$ likelihood 面的实验，不能直接换成 $1+3+1$。

---

## 3. MicroBooNE

### 3.1 官方分析使用什么

MicroBooNE 2025 双束流论文同时使用 BNB 和 NuMI，并将每束流分为七个通道：

1. $\nu_e$ CC FC；
2. $\nu_e$ CC PC；
3. $\nu_\mu$ CC FC；
4. $\nu_\mu$ CC PC；
5. $\nu_\mu$ CC $\pi^0$ FC；
6. $\nu_\mu$ CC $\pi^0$ PC；
7. NC $\pi^0$。

CC 表示带电流相互作用，NC 表示中性流相互作用，FC 表示最终态完全包含在探测器内，PC 表示部分包含。每个通道含25个 $0$--$2.5\,\mathrm{GeV}$、宽度为 $0.1\,\mathrm{GeV}$ 的重建能量 bin，以及一个 $>2.5\,\mathrm{GeV}$ overflow bin。因此官方联合向量共有

$$
14\times26=364
$$

个 bin。

官方使用全部14个样本和完整 $364\times364$ 协方差。论文强调条件约束通过完整协方差的相关项自动进入联合 $\chi^2$，并用伪实验得到 $CL_s$ 排除线。论文地址：

- [MicroBooNE 2025 Nature论文](https://www.nature.com/articles/s41586-025-09757-7)
- 重点位置：Fig. 1（束流及参考振荡谱）、Fig. 2（constrained谱）、Fig. 3（参数限制）、Methods “Statistical methods for oscillation analysis”、Extended Data Fig. 2和3。

### 3.2 当前仓库读取的公开输入

公开输入位于：

```text
data/experiments/microboone/shared/raw/hepdata_microboone_2025/
```

包含：

- `HEPData-ins3088922-v1-Unconstrained_14_channels.csv`：全部364个 bin 的 Data、Background、Signal+Background及数据统计误差；
- `HEPData-ins3088922-v1-14_channel_covariance_matrix.csv`：全部 $364\times364$ 系统协方差。

当前正式数值分析只选取每束流前四个 CC 通道：

$$
\nu_e\ \mathrm{CC\ FC},\quad
\nu_e\ \mathrm{CC\ PC},\quad
\nu_\mu\ \mathrm{CC\ FC},\quad
\nu_\mu\ \mathrm{CC\ PC}.
$$

因此：

- BNB：$4\times26=104$ bins；
- NuMI：$4\times26=104$ bins；
- 联合：$208$ bins。

程序从完整协方差中同时抽取

$$
\Sigma_{\rm syst}^{\rm selected}
=
\begin{pmatrix}
\Sigma_{\rm BNB}&\Sigma_{\rm BNB,NuMI}\\
\Sigma_{\rm NuMI,BNB}&\Sigma_{\rm NuMI}
\end{pmatrix},
$$

包括 BNB–NuMI 的两个交叉块。遗漏的六个 $\pi^0$/NC 通道不会参与条件约束，这是当前分析与官方14通道结果的重要差异。

### 3.3 BNB真能量输入与Reco

BNB输入通量位于：

```text
data/experiments/microboone/bnb/inputs/bnb_flux.csv
```

它包含60个真能量 bin、$0$--$3\,\mathrm{GeV}$、bin宽 $0.05\,\mathrm{GeV}$ 的四种束流成分：

$$
\Phi_{\nu_\mu}(E),\quad
\Phi_{\bar\nu_\mu}(E),\quad
\Phi_{\nu_e}(E),\quad
\Phi_{\bar\nu_e}(E).
$$

Reco输入来自较早的 MicroBooNE 2022 HEPData发布：

```text
data/experiments/microboone/bnb/raw_response/
```

原始矩阵被转换为

$$
R_{rt}=P(E_{\rm reco}\in r\mid E_{\rm true}\in t).
$$

每个真能量列归一化到1；完全为零的列保持全零。随后将其适配为每通道 $26\times60$ 的矩阵。该 Reco 来自较早发布，不能视为2025分析内部 detector response 的直接公开版本。

### 3.4 BNB经验kernel如何构造

HEPData 的 `Signal + Background` 当前被当作零惰性混合的经验参考总谱：

$$
M_r^{\rm ref}=S_r^{\rm ref}+B_r^{\rm pub},
$$

其中：

- $B_r^{\rm pub}$ 是 HEPData 的 Background；
- $S_r^{\rm ref}=M_r^{\rm ref}-B_r^{\rm pub}$ 是由两个公开列相减得到的参考 signal；
- 上标 `ref` 表示构造 kernel 所使用的参考点。

当前配置把参考混合设为零：

$$
s_{14}=s_{24}=0.
$$

对给定末态通道，先计算

$$
Q_r^{\rm ref}
=\sum_{\alpha,t}
R_{rt}\Phi_{\alpha t}
P_{\alpha\rightarrow\beta}(E_t,L;\boldsymbol\eta_{\rm ref}),
$$

再定义每个重建 bin 的比例

$$
c_r=\frac{S_r^{\rm ref}}{Q_r^{\rm ref}}.
$$

最终保存的过程矩阵是

$$
K_{rt}^{\alpha\rightarrow\beta}
=c_rR_{rt}\Phi_{\alpha t}.
$$

扫描时使用

$$
M_r(\boldsymbol\eta)
=B_r^{\rm pub}
+\sum_{\alpha,\beta,t}
K_{rt}^{\alpha\rightarrow\beta}
P_{\alpha\rightarrow\beta}(E_t,L;\boldsymbol\eta).
$$

代码保存8个 CC 过程矩阵：

$$
\nu_e\to\nu_e,
\quad\nu_\mu\to\nu_e,
\quad\nu_e\to\nu_\mu,
\quad\nu_\mu\to\nu_\mu,
$$

以及四个对应反中微子过程。参考点下必须逐 bin 精确闭合 HEPData 的 `Signal + Background`。

> [!warning] BNB kernel的物理限制
> 逐重建-bin比例 $c_r$ 吸收了未知截面、效率、归一化和选择效应，但这并不等价于知道这些量的真实能量依赖。
> 同一末态内的不同初始味道以及中微子/反中微子共享该有效比例。
> 更严重的是，HEPData没有明确证明当前 `Unconstrained Signal + Background` 就是论文用于检验的3ν/null总预测，因此“零混合锚点”是当前工程假设，而不是官方事实。

### 3.5 NuMI通量和经验kernel

NuMI原始输入分为两种磁角运行模式：

- FHC：Forward Horn Current，正向聚焦设置；
- RHC：Reverse Horn Current，反向聚焦设置。

每种模式保存

$$
\nu_\mu,\quad\bar\nu_\mu,\quad\nu_e,\quad\bar\nu_e
$$

四种成分，总计8份通量。输入位于：

```text
data/experiments/microboone/numi/inputs/flux_components/
```

这8份数组不是官方机器可读 flux release，而是从 MicroBooNE 技术报告 PDF 的矢量路径恢复，并通过重绘叠加进行人眼检查。网格为 $0$--$5\,\mathrm{GeV}$、50个 $0.1\,\mathrm{GeV}$ bins。

程序按当前记录的曝光比例组合：

$$
\Phi_\alpha^{\rm combined}(E)
=0.308\Phi_\alpha^{\rm FHC}(E)
+0.692\Phi_\alpha^{\rm RHC}(E).
$$

NuMI事件 kernel 暂时复用 BNB 的 $26\times60$ Reco，并把 $0.1\,\mathrm{GeV}$ flux 转到 BNB 的 $0.05\,\mathrm{GeV}$ 真能量网格。因为 Reco 只覆盖 $0$--$3\,\mathrm{GeV}$，NuMI flux 的 $3$--$5\,\mathrm{GeV}$ 部分没有进入事件预测。

随后用与 BNB 相同的逐重建-bin闭合方法构造8个过程矩阵，使参考点下重建总谱严格等于 NuMI 四通道的公开 `Signal + Background`。

NuMI使用单基线

$$
L_{\rm NuMI}=0.680\,\mathrm{km},
$$

没有积分 NuMI 衰变管中不同产生位置形成的真实基线分布。

> [!warning] NuMI相对BNB新增的限制
> NuMI除了继承经验锚点和固定聚合背景问题，还存在PDF flux恢复误差、借用BNB Reco、缺少3--5 GeV响应、单基线近似。因而当前NuMI可用于近似联合研究，不能称为官方NuMI detector model。

### 3.6 固定背景和可振荡成分

当前代码把 HEPData 整个 Background 列固定为

$$
B_r^{\rm fixed}=B_r^{\rm pub}.
$$

只有从 `Signal = Signal+Background-Background` 重建出的部分被拆成 $e/\mu$、中微子/反中微子过程并重加权。

这意味着 Background 内即使包含内禀 $\nu_e$、错辨 $\nu_\mu$、NC或其他原则上会受惰性振荡影响的中微子成分，在当前实现中也不会改变。这不是一般物理要求，而是因为公开表没有提供足够的 component-level 真能量模板。

### 3.7 当前协方差和卡方

令：

- $\mathbf D$ 为选中208个 bin 的观测列向量；
- $\mathbf M(\boldsymbol\eta)$ 为对应预测列向量；
- $\Sigma_{\rm syst}^{\rm ref}$ 为从官方364维系统协方差抽取的参考子矩阵；
- $M_i^{\rm ref}$ 为第 $i$ 个 bin 的参考预测；
- $M_i(\boldsymbol\eta)$ 为当前参数点的预测。

当前代码把整个系统协方差按总预测比例缩放：

$$
[\Sigma_{\rm syst}(\boldsymbol\eta)]_{ij}
=
[\Sigma_{\rm syst}^{\rm ref}]_{ij}
\frac{M_i(\boldsymbol\eta)}{M_i^{\rm ref}}
\frac{M_j(\boldsymbol\eta)}{M_j^{\rm ref}}.
$$

统计协方差采用论文 Methods 所述 Pearson 形式：

$$
[\Sigma_{\rm stat}(\boldsymbol\eta)]_{ij}
=\delta_{ij}M_i(\boldsymbol\eta),
$$

其中 $\delta_{ij}$ 是 Kronecker delta：当 $i=j$ 时为1，否则为0。总协方差是

$$
\Sigma(\boldsymbol\eta)
=\Sigma_{\rm syst}(\boldsymbol\eta)
+\Sigma_{\rm stat}(\boldsymbol\eta).
$$

卡方为

$$
\chi^2(\boldsymbol\eta)
=
[\mathbf D-\mathbf M(\boldsymbol\eta)]^T
\Sigma^{-1}(\boldsymbol\eta)
[\mathbf D-\mathbf M(\boldsymbol\eta)].
$$

程序不显式计算逆矩阵，而是对协方差作 Cholesky 分解并求解三角方程；这与上述二次型代数等价。

官方论文说明：振荡相关的中微子相互作用保持分数系统误差不变，而非中微子背景和fiducial volume之外的中微子背景保持绝对误差不变。公开协方差没有把这些成分拆开，所以当前代码把整个矩阵一起按总预测比例缩放，是一个额外近似。

### 3.8 Fig. 3a profile

Fig. 3a 扫描坐标为

$$
\left(\Delta m_{41}^2,\sin^2(2\theta_{\mu e})\right).
$$

当前 preset 范围是

$$
10^{-2}\leq\Delta m_{41}^2\leq10^2\ \mathrm{eV}^2,
$$

$$
10^{-4}\leq\sin^2(2\theta_{\mu e})\leq1.
$$

每个二维网格点固定 $\Delta m_{41}^2$ 和 $A_{\mu e}\equiv\sin^2(2\theta_{\mu e})$。由于

$$
A_{\mu e}=4s_{14}(1-s_{14})s_{24},
$$

程序沿完整物理区间 profile $s_{14}$，并由约束推导

$$
s_{24}=\frac{A_{\mu e}}{4s_{14}(1-s_{14})}.
$$

允许区间由 $0\leq s_{24}\leq1$ 决定。代码先使用33点确定性盆地搜索，再对发现的局部盆地作有界精细最小化。它不是固定 $s_{14}=s_{24}$，也不是只取小角分支。

### 3.9 Fig. 3b profile

Fig. 3b 扫描坐标为

$$
\left(\Delta m_{41}^2,\sin^2(2\theta_{ee})\right).
$$

当前 preset 范围是

$$
10^{-1}\leq\Delta m_{41}^2\leq14\ \mathrm{eV}^2,
$$

$$
10^{-2}\leq\sin^2(2\theta_{ee})\leq1.
$$

令 $A_{ee}\equiv\sin^2(2\theta_{ee})$。固定 $A_{ee}$ 后，$s_{14}$ 有两个物理解：

$$
s_{14}^{\pm}
=\frac{1\pm\sqrt{1-A_{ee}}}{2}.
$$

程序在每个分支上分别对

$$
0\leq s_{24}\leq1
$$

进行有界 profile，最后保留两个分支中较小的 $\chi^2$。

### 3.10 当前 $CL_s$ 与无Toy方法

在每个扫描点，令：

- $H_3$：零惰性混合的 $3\nu$ 假设；
- $H_4$：该二维点经过观测数据 profile 后的 $4\nu$ 假设；
- $\chi^2_3$：在 $H_3$ 下得到的卡方；
- $\chi^2_4$：在 $H_4$ 下得到的卡方。

检验统计量定义为

$$
T=\chi^2_4-\chi^2_3.
$$

当前右尾概率定义为

$$
p_4=P(T_{\rm pseudo}\geq T_{\rm obs}\mid H_4),
$$

$$
p_3=P(T_{\rm pseudo}\geq T_{\rm obs}\mid H_3),
$$

其中 $T_{\rm obs}$ 是真实观测数据给出的检验量，$T_{\rm pseudo}$ 是假想重复实验中的检验量。随后定义

$$
CL_s=\frac{p_4}{p_3}.
$$

当

$$
CL_s\leq0.05
$$

时，该点标记为95% $CL_s$ 排除。

当前 `--calibration analytic` **不是正态近似**。它首先把两套预测和两套协方差固定，然后利用多元高斯变量下两个二次型之差的广义二次型分布。程序通过矩阵特征值和线性项建立该分布，并数值反演其特征函数，分别直接计算 $H_3$ 与 $H_4$ 下的右尾概率。旧的只保留均值和方差的 Gaussian 方法仍可在研究脚本中作为诊断对照，但不决定当前主扫描排除线。

### 3.11 当前Toy方法

`--calibration toy` 时，在每个扫描点执行：

1. 对真实观测数据执行上述 Fig. 3a 或 Fig. 3b profile，得到该点的 $H_4$ 参数；
2. 构造 $H_3$ 与 $H_4$ 各自的预测均值和预测相关协方差；
3. 分别生成
   $$
   \mathbf X_H=\boldsymbol\mu_H+L_H\mathbf z,
   \qquad\mathbf z\sim\mathcal N(\mathbf0,I),
   $$
   其中 $\mathbf X_H$ 是假设 $H$ 下的一份伪数据，$\boldsymbol\mu_H$ 是该假设的预测均值，$L_HL_H^T=\Sigma_H$ 是协方差的 Cholesky 分解，$\mathbf z$ 是标准多元正态随机向量；
4. 对每份伪数据计算固定 $H_4$ 和固定 $H_3$ 的两个二次型之差；
5. 统计超过 $T_{\rm obs}$ 的右尾数量；
6. 使用有限样本修正
   $$
   \widehat p=\frac{n_{\rm tail}+1}{N_{\rm toy}+1},
   $$
   其中 $n_{\rm tail}$ 是尾部Toy数，$N_{\rm toy}$ 是每个假设生成的Toy总数。

> [!important] 当前Toy不做什么
> 当前活动主程序不会在每一份Toy内部重新profile振荡参数。它是“观测数据每点profile一次，然后固定两套假设进行Toy校准”的plug-in方法。
> 先前存在的逐Toy-profile研究只位于 `studies/`，不能用来描述当前正式扫描。

`--calibration adaptive-toy` 先对全网格计算广义二次型 $CL_s$，再只对指定 $CL_s$ 区间及邻近点运行Toy；其余网格保留解析值。因此它是一张显式混合结果，不是全网格Toy结果。

### 3.12 当前与官方MicroBooNE的差异

| 环节 | 官方分析 | 当前仓库 |
|---|---|---|
| 通道 | BNB+NuMI全部14通道、364 bins | 每束流前4个CC通道，共208 bins |
| 探测器预测 | 合作组内部MC、分类成分和响应 | 从公开总谱反推的经验kernel |
| BNB响应 | 2025内部分析响应 | 借用较早2022 Reco并适配 |
| NuMI flux | 内部束流模拟 | 从技术报告PDF矢量图恢复 |
| NuMI响应 | NuMI对应内部模拟 | 借用BNB Reco，且没有3--5 GeV支持 |
| 背景振荡 | 按物理成分更新 | 整个公开Background固定 |
| 系统协方差更新 | 振荡相关和固定成分分别处理 | 整个系统矩阵按总预测比例缩放 |
| profile | 二维平面上profile第三个4ν参数 | 使用精确 $s_{14},s_{24}$ 参数化实现相同目标 |
| $CL_s$ | 使用完整协方差生成伪实验 | analytic为广义二次型精确反演；Toy为固定profile点的plug-in伪实验 |
| 每Toy重新profile | 论文公开文字未充分说明内部每Toy处理细节 | 正式代码不重新profile |

因此当前谱形和参数面的大体一致只能说明经验映射抓住了主要趋势；不能用来证明官方数据生成、系统误差处理和覆盖率已被严格复现。

### 3.13 MicroBooNE接入 $1+3+1$

当前 $1+3+1$ 只接入 MicroBooNE BNB 或近似 BNB+NuMI kernel。质量平面固定：

$$
|\Delta m_{41}^2|,
\qquad
\Delta m_{51}^2,
$$

并 profile：

$$
|U_{e4}|^2,
|U_{\mu4}|^2,
|U_{e5}|^2,
|U_{\mu5}|^2,
\phi_{\mu e}.
$$

优化器同时检查：

- 完整五维混合体积；
- 状态4退耦边界；
- 状态5退耦边界；
- 两个重态都退耦的 $3\nu$ 边界。

默认只是 $7\times7$ 质量网格，属于开发诊断。解析校准仍用固定假设广义二次型；Toy仍为观测数据profile后固定点的Toy，不在每个Toy中重新profile。

与理想统一 $1+3+1$ 实验参照相比，当前最大问题不是概率公式，而是 kernel 不能确认所有可振荡背景和NC成分均已分解。因此它可以作为条件性模型替换研究，尚不能称为完整MicroBooNE $1+3+1$ 限制。

### 3.14 MicroBooNE建议插图

本仓库已有图：

```text
outputs/microboone_bnb_numi_joint/three_plus_one/layout_check_20260903/spectra_figure1/figure1_nue_cc_fc.png
outputs/microboone_bnb_numi_joint/three_plus_one/layout_check_20260903/contour_comparison/
```

建议从官方论文查找：Fig. 1、Fig. 2、Fig. 3、Extended Data Fig. 2、Extended Data Fig. 3，以及 Methods 的协方差和 $CL_s$ 公式。

---

## 4. MiniBooNE

### 4.1 当前读取的官方发布数据

输入位于：

```text
data/experiments/miniboone/shared/raw/official_nue2020_combined/
```

使用的是 MiniBooNE 2020/2021 $\nu+\bar\nu$ 联合 appearance release。主要输入包括：

- 中微子模式电子样本：11个重建能量 bins 的 Data 和 Background；
- 中微子模式缪子控制样本：8个 bins 的 Data 和预测；
- 反中微子模式电子样本：11个 bins 的 Data 和 Background；
- 反中微子模式缪子控制样本：8个 bins 的 Data 和预测；
- 中微子模式 full-transmutation MC：17204行；
- 反中微子模式 full-transmutation MC：117949行；
- $60\times60$ 分数协方差矩阵；
- 官方 $190\times190$ likelihood surface；
- 官方 $1\sigma$、90%、99%、$3\sigma$ 轮廓。

论文来源：

- [MiniBooNE更新结果，arXiv:2006.16883](https://arxiv.org/abs/2006.16883)
- 输入来源和逐文件哈希见 `data/experiments/miniboone/README.md` 与 `SHA256SUMS.csv`。

### 4.2 逐事件振荡重加权

每个 full-transmutation MC 行包含：

- $E_{\rm QE}$：用于落入11个电子样本 bin 的重建准弹性能量；
- $E_{\rm true}$：真实中微子能量；
- $L_{\rm true}$：真实传播距离；
- $w$：官方事件权重；
- $N_{\rm MC}$：对应 full-transmutation 表的总行数。

对每个事件 $k$，代码调用公共 $3+1$ 内核计算

$$
P_k=P_{\mu e}(E_{{\rm true},k},L_{{\rm true},k};\Delta m_{41}^2,A_{\mu e}),
$$

其中 $A_{\mu e}=\sin^2(2\theta_{\mu e})$。第 $r$ 个电子样本 bin 的振荡信号为

$$
S_r
=\sum_{k\in r}\frac{P_kw_k}{N_{\rm MC}}.
$$

这是官方发布说明给出的 $P\,w/N$ 规则。代码为了调用统一 $3+1$ 核心，选取 appearance 简并族中的代表参数

$$
s_{14}=\frac12,
\qquad
s_{24}=A_{\mu e},
$$

使

$$
4s_{14}(1-s_{14})s_{24}=A_{\mu e}.
$$

该代表选择只保证 $P_{\mu e}$ 完全正确；不能用它解释 $P_{ee}$ 或 $P_{\mu\mu}$。

### 4.3 MiniBooNE预测和协方差

官方60维分数协方差的六个块按以下顺序排列：

```text
ν-mode signal (11)
ν-mode electron-like background (11)
ν-mode muon control (8)
anti-ν-mode signal (11)
anti-ν-mode electron-like background (11)
anti-ν-mode muon control (8)
```

令 $f_{ij}$ 是发布的分数协方差，$q_i$ 是当前六块预测，则绝对协方差先构造为

$$
V_{ij}^{60}=f_{ij}q_iq_j.
$$

随后使用一个固定线性折叠矩阵 $C$，把 signal 与对应电子背景相加：

$$
V^{38}=CV^{60}C^T.
$$

最终38维顺序是

$$
(11\ \nu_e\text{-like},\ 8\ \nu_\mu,
11\ \bar\nu_e\text{-like},\ 8\ \bar\nu_\mu).
$$

发布矩阵包含信号系统误差但不包含 full-transmutation 信号的统计误差，因此代码在两个电子信号块的对角线上分别加 $S_r$。

当前本地 Gaussian NLL 为

$$
-2\ln L
=
(\mathbf D-\mathbf M)^TV^{-1}(\mathbf D-\mathbf M)
+\ln\det V.
$$

这里加入 $\ln\det V$ 是因为 $V$ 随振荡信号变化；如果只保留二次型，会遗漏参数相关高斯归一化。

### 4.4 MiniBooNE扫描、profile和Toy

本地扫描直接遍历官方二维网格：

$$
(\Delta m_{41}^2,\sin^2(2\theta_{\mu e})).
$$

没有额外的 $s_{14}$/$s_{24}$ profile；二维 appearance release 本身无法区分二者。程序比较：

- 本地重建的 Gaussian NLL 面；
- 官方发布的 $190\times190$ likelihood 面；
- 官方发布的频率学派校准轮廓。

当前 MiniBooNE 本地 `scan` **没有运行Toy MC**，也没有重新生成官方覆盖率轮廓。图上的官方轮廓只是读取官方文本文件进行对照，不是本地Toy结果。

### 4.5 与官方MiniBooNE的差异

当前实现高度依赖官方逐事件信号样本，因此比从图片或单一能谱反推更接近官方两味 appearance 分析。但仍有以下边界：

- 本地使用公开说明重建 Gaussian NLL，不等于拥有合作组所有内部 likelihood 细节；
- 本地没有复现官方频率学派 coverage calibration；
- 背景和缪子控制样本按公开包的方式固定，没有把它们拆成完整 $3+1$ survival/appearance 成分；
- 本地二维参数只描述 $P_{\mu e}$，没有同时扫描 appearance 与 disappearance。

### 4.6 MiniBooNE接入 $1+3+1$ 时的差异

full-transmutation MC 含逐事件 $E_{\rm true}$ 和 $L_{\rm true}$，所以理论上可以把每个事件的

$$
P_{\mu e}^{3+1}(E,L)
$$

替换为

$$
P_{\mu e}^{1+3+1}(E,L),
$$

并保留两种频率和CP干涉。这部分比只拥有二维官方面更适合模型替换。

但是，公开包没有把全部电子背景、缪子控制样本和内禀中微子成分按初始味道和真能量完整分解。因此当前数据足以建立 **appearance-only $1+3+1$ 项**，但不足以自动建立和 MicroBooNE 一样同时重加权 $P_{ee}$、$P_{\mu e}$、$P_{e\mu}$、$P_{\mu\mu}$ 的完整项。当前代码尚未实现这一替换。

### 4.7 MiniBooNE建议插图

本仓库已有：

```text
outputs/miniboone_nu_nubar_combined/two_flavour/miniboone_3plus1_reconstruction_20260905/scan/parameter_space.png
outputs/miniboone_nu_nubar_combined/two_flavour/miniboone_3plus1_reconstruction_20260905/scan/parameter_space_line_overlay.png
```

建议从官方论文及数据页查找电子样本谱、允许区和频率学派轮廓说明。

---

## 5. LSND

### 5.1 当前实现的定位与文件

LSND适配器位于：

```text
src/sterile_fit/experiments/lsnd/final_2001.py
src/sterile_fit/experiments/lsnd/public_rate_approximation.py
src/sterile_fit/experiments/lsnd/adapter.py
```

公开输入及来源记录位于：

```text
data/experiments/lsnd/published/final_2001_summary.csv
data/experiments/lsnd/sources/MANIFEST.csv
data/experiments/lsnd/sources/aguilar_2001_hep-ex_0104049.pdf
```

当前实现是 **DAR appearance总率近似**。它的目的不是反演LSND探测器，而是在没有事件表和官方likelihood数组时，建立一个明确依赖

$$
P_{\bar\nu_\mu\rightarrow\bar\nu_e}(E,L)
$$

的最低信息量实验项，使同一实验权重以后可以接收 $3+1$ 或 $1+3+1$ 概率。它不是LSND官方分析，也没有证据表明Kopp等全球拟合作者使用了这条一维Gaussian总率式。

### 5.2 每个输入参数的来源和作用

| 程序量 | 当前数值 | 来源 | 在计算中的作用 |
|---|---:|---|---|
| $P_{\rm obs}$ | $0.00264$ | LSND最终论文摘要、Sec. VI和Table XI | 总率likelihood的观测中心 |
| $\sigma_{\rm stat}$ | $0.00067$ | 同一结果的第一项误差 | 观测概率的统计宽度 |
| $\sigma_{\rm syst}$ | $0.00045$ | 同一结果的第二项误差 | 观测概率的系统宽度 |
| $L_{\rm centre}$ | $30.0\,\mathrm m$ | LSND Sec. II.E | 源到探测器中心的基线 |
| $\ell_{\rm det}$ | $8.3\,\mathrm m$ | LSND Sec. II.E | 当前轴向基线平均的范围 |
| $E_e^{\rm cut}$ | $20\,\mathrm{MeV}$ | LSND主DAR选择 $20<E_e<60\,\mathrm{MeV}$ | 换算积分的最低中微子能量 |
| $E_{\rm max}$ | $52.8\,\mathrm{MeV}$ | 静止 $\mu^+$ 三体衰变运动学；LSND Sec. II.B | Michel谱端点 |
| $\Delta=m_n-m_p$ | $1.293332\,\mathrm{MeV}$ | 标准粒子质量差 | $E_e\simeq E_\nu-\Delta$ |
| $m_e$ | $0.510999\,\mathrm{MeV}$ | 标准电子质量 | $p_e=\sqrt{E_e^2-m_e^2}$ |
| 能量积分点 | 512 | 数值设置，不是实验数据 | 对连续DAR谱积分 |
| 基线积分点 | 129 | 数值设置，不是实验数据 | 对简化的一维探测器长度积分 |

LSND公布

$$
P_{\rm obs}=(0.264\pm0.067\pm0.045)\%,
$$

是根据观测超额、预测的完全转换事件数、束流、截面和效率得到的 **样本平均转换概率**。它不是某个能量bin的概率，也不是振荡公式中的振幅 $\sin^2(2\theta_{\mu e})$。程序将两项误差视为独立并合并：

$$
\sigma_P=\sqrt{\sigma_{\rm stat}^2+\sigma_{\rm syst}^2}.
$$

因此 $P_{\rm obs}$ 的作用只是告诉程序：经过LSND事件权重平均后的理论概率 $\overline P$ 应接近多少。它没有提供事件在 $E$、$L$、$R_\gamma$、位置或方向上的分布。

LSND说明系统误差包含背景、7%的DAR flux、7%的正电子效率和7%的俘获光子效率。当前代码没有再次分别加入这些误差，因为它们已经包含在公开的 $\sigma_{\rm syst}$ 中；再次加入会双计数。[LSND最终论文](https://arxiv.org/abs/hep-ex/0104049)

### 5.3 Michel谱为什么出现

信号源链为

$$
\pi^+\rightarrow\mu^++\nu_\mu,
\qquad
\mu^+\rightarrow e^++\nu_e+\bar\nu_\mu.
$$

绝大多数 $\mu^+$ 停止后衰变，所以信号初态 $\bar\nu_\mu$ 的能谱由静止μ子三体衰变运动学决定。忽略很小的辐射和极化修正，当前使用归一化Michel形状

$$
f_{\bar\nu_\mu}(x)=2x^2(3-2x),
\qquad
x=\frac{E_\nu}{E_{\rm max}},
\qquad0\le x\le1.
$$

LSND明确说明停止的 $\mu^+$ 产生正常Michel谱，并用完整production MC按年份、靶站和探测器内25个位置计算实际flux。当前公式只恢复主要的DAR能量形状；它没有恢复A1/A2小贡献、A6源尺寸、不同年份靶构型或25位置flux表。[LSND Sec. II.B--D](https://arxiv.org/abs/hep-ex/0104049)

### 5.4 IBD截面权重为什么出现

LSND的主要DAR appearance信号是逆β衰变（IBD）：

$$
\bar\nu_e+p\rightarrow e^++n.
$$

振荡概率相同的两个能量区间并不会产生同样多的事件，因为IBD截面随能量快速增加。最低阶总截面的主要形状为

$$
\sigma_{\rm IBD}(E_\nu)
\simeq0.0952\times10^{-42}
\left(\frac{E_ep_e}{\mathrm{MeV}^2}\right)\mathrm{cm}^2,
$$

$$
E_e\simeq E_\nu-(m_n-m_p),
\qquad
p_e=\sqrt{E_e^2-m_e^2}.
$$

该 $E_ep_e$ 主导形式来自低能IBD标准计算；更精确公式还包含反冲、弱磁、角分布和辐射修正。[Vogel--Beacom, Phys. Rev. D 60, 053003](https://doi.org/10.1103/PhysRevD.60.053003)

当前likelihood比较的是已经归一化的平均概率，所以截面的绝对常数、总质子数、总POT和flux绝对归一化在权重分子与分母中约掉；真正保留下来的是截面随能量的相对形状。因此代码只写 $\sigma\propto E_ep_e$。这不表示LSND官方只用了最低阶截面；官方使用完整MC及其系统误差。

### 5.5 当前等效事件权重与模型概率

当前构造

$$
w(E,L)=
f_{\bar\nu_\mu}(E)\,
\sigma_{\rm IBD}(E)\,
\epsilon(E,L)\,
G(L),
$$

其中 $\epsilon$ 是选择效率，$G$ 是几何flux因子。当前近似取

$$
\epsilon(E,L)=\mathrm{constant},
\qquad
G(L)\propto\frac1{L^2},
$$

并把圆柱的8.3 m轴向长度简化为

$$
L\in[30-4.15,30+4.15]\,\mathrm m
$$

上的均匀积分。于是

$$
\overline P(\boldsymbol\eta)=
\frac{\int dE\,dL\;w(E,L)
P_{\bar\mu\bar e}(E,L;\boldsymbol\eta)}
{\int dE\,dL\;w(E,L)}.
$$

$\boldsymbol\eta$ 表示振荡模型参数。对 $3+1$，

$$
P_{\bar\mu\bar e}^{3+1}
=\sin^2(2\theta_{\mu e})
\sin^2\!\left(1.267\frac{\Delta m^2_{41}L}{E}\right).
$$

程序在512个 $E$ 点和129个 $L$ 点分别计算概率，不把不同能量视作同一个概率。恒定效率会在归一化比值中消掉；真实的能量、位置和 $R_\gamma$ 依赖效率不会消掉，因此这是当前最重要的未建模项之一。

### 5.6 为什么不用论文里公开的粗flux曲线

LSND最终论文Table III公开的是不同来源和年份的 **积分flux**，不是可直接读取的逐能量bin数组；Fig. 3给出能谱曲线，但公开PDF图并不是带误差、bin边界和位置依赖的机器可读数据产品。对于纯 $\mu^+$ DAR主成分，Michel解析形状由衰变运动学确定，直接使用它通常比从旧图像读取少量点更透明，也避免人为数字化误差。

但是Michel谱不能替代LSND production MC。官方MC还包含：

- A1、A2和A6源的空间与年份权重；
- 每年不同靶构型；
- 探测器内25个位置的flux；
- DIF成分及小的内禀 $\bar\nu_e$ 背景；
- 真实几何、效率、能量分辨和事件选择。

因此“不使用粗图”不等于信息已经完整，而是选择一个可审计的解析近似。若以后取得官方数值flux表，应直接替换Michel+简化几何权重；不应把PDF曲线数字化结果称为官方数值输入。

### 5.7 当前统计方法和扫描

当前只有一个观测量，使用一维Gaussian总率likelihood：

$$
-2\ln L_{\rm rate}(\boldsymbol\eta)
=\frac{[\overline P(\boldsymbol\eta)-P_{\rm obs}]^2}{\sigma_P^2}.
$$

当前 $3+1$ 扫描为 $241\times241$：

$$
(\Delta m^2_{41},\sin^2(2\theta_{\mu e})).
$$

由于 $3+1$ appearance概率对振幅严格线性，程序对每个质量差计算一次单位振幅平均，再乘以不同振幅；这是代数等价加速。绘图使用相对网格最小值的恒定切片

$$
\Delta[-2\ln L]=4.605,\qquad9.210,
$$

作为二维90%和99%参考线。当前没有nuisance profile、Toy MC或 $CL_s$，也没有把官方允许区作为拟合输入。

### 5.8 LSND官方实际上怎样做

LSND最终振荡结果不是总率拟合。官方对 $20<E_e<200\,\mathrm{MeV}$ 的5697个beam-on事件建立四维扩展likelihood，使用

$$
(E_e,R_\gamma,\cos\theta_\nu,z).
$$

这里 $E_e$ 是重建电子/正电子能量；$R_\gamma$ 衡量延迟2.2 MeV中子俘获光子与主事件的关联；$\cos\theta_\nu$ 是相对束流方向；$z$ 是轴向位置。信号和多类背景都有自己的MC PDF及归一化约束。官方在每个振荡点用

$$
\sin^2(1.27\Delta m^2L_\nu/E_\nu)
$$

对MC事件重加权，并对有限的 $L_\nu$、$E_\nu$ 分辨进行smearing。最终联合DAR appearance、DIF appearance和已知背景，得到最佳点

$$
(\sin^22\theta,\Delta m^2)=(0.003,1.2\,\mathrm{eV}^2).
$$

所以官方使用Michel谱、截面、效率和几何只是完整事件生成链的一部分；官方不是把 $P_{\rm obs}$ 塞进一维Gaussian式来获得Fig. 27。[LSND Sec. IX.G--H](https://arxiv.org/abs/hep-ex/0104049)

### 5.9 Kopp等 $1+3+1$ 全球拟合怎样使用LSND

Kopp、Machado、Maltoni和Schwetz把LSND作为11个自由度的 $\bar\nu_\mu\rightarrow\bar\nu_e$ appearance项，与MiniBooNE、KARMEN等appearance项及各disappearance项相加。他们在同一套全局参数下评价LSND项，因此在 $1+3+1$ 中替换的是包含两个频率和CP干涉的

$$
P_{\bar\mu\bar e}^{1+3+1}(E,L),
$$

而不是把LSND的二维 $3+1$ 轮廓当成可直接推广的函数。[Kopp et al., JHEP 05 (2013) 050](https://arxiv.org/abs/1303.3011)

但该论文没有公开足够信息来证明他们使用了当前代码的Michel×最低阶IBD×均匀轴向总率积分。论文也没有给出可从头重建其LSND 11-dof项的事件表、bin表或代码。其方法链条中的较早全球分析明确写道：LSND合作组向作者提供了“LSND global”和“LSND DAR”两套分析所得的likelihood函数，作者再按 $\chi^2\propto-2\ln\mathcal L$ 转换；前者来自5697事件的 $20<E_e<200$ MeV DAR+DIF样本，后者来自以DAR为主的1032事件样本。这证明全球拟合使用的信息量高于一个公开平均概率，也说明外部读者仅凭论文图表无法重建相同输入。[Maltoni et al., Sec. 5](https://arxiv.org/abs/hep-ph/0207157)

Kopp 2013表中“LSND 11 dof”说明其全局目标函数中LSND appearance项贡献的有效数据自由度，但并不等于公开了11个bin的数值、响应或协方差；不能据此自行假定是哪11个能量bin。因此必须区分：

| 分析 | LSND数据层级 | 能否复现官方形状 | 能否自然换成 $1+3+1$ |
|---|---|---|---|
| LSND官方 | 四维事件likelihood、MC PDF、DAR+DIF | 是 | 原理上能，但需要内部输入和重新拟合 |
| Kopp等全球拟合 | 11-dof reduced LSND appearance项；实现细节未完整公开 | 接近其采用的LSND输入 | 是；他们已在统一全局参数中使用 |
| 当前仓库 | 一个平均概率的DAR总率Gaussian项 | 否，只约束平均振荡强度 | 可以替换概率，但只有rate-only约束 |

因此当前LSND代码是一个独立的保守近似，不是“Kopp方法的复刻”。

### 5.10 换成 $1+3+1$ 时当前信息能做什么

`average_appearance_probability` 接受任意概率函数，所以可以直接将 $3+1$ 概率换成

$$
P_{\bar\mu\bar e}^{1+3+1}(E,L),
$$

从而保留两个质量频率、$\Delta m^2_{54}$ 干涉和反中微子CP相位符号。实验权重仍是相同的Michel×IBD×简化几何；这正是跨模型重加权应保持不变的部分。

但一个 $P_{\rm obs}$ 只能约束一个加权平均数。不同质量对、四个混合模和相位可能产生相同 $\overline P$，所以它不能恢复LSND的 $L/E$ 形状辨别力。该项可以进入标明近似的条件性全局分析，不能宣称复现Kopp或LSND官方的 $1+3+1$ likelihood。

### 5.11 网络公开信息还能提升到什么程度

截至本次检索，可确认公开网络上有：LSND最终论文、积分flux表、论文中的能谱和 $L/E$ 投影、历史LSND/LSU flux说明页面、若干截面论文和后来全球拟合结果。没有找到经过合作组维护、机器可读且同时包含以下内容的最终公开包：

- 5697个事件的四变量表；
- 各信号和背景的四维PDF；
- 逐位置、逐年份的DAR/DIF数值flux；
- 完全转换MC或 $E_{\rm true},L_{\rm true}\rightarrow$ 重建变量响应；
- 官方二维/多维数值log-likelihood网格。

可实现的提升按价值排序为：

1. 向LSND作者或保存全球拟合代码的作者索取最终appearance likelihood/11-dof输入；这是唯一能显著接近Kopp层级的路径。
2. 若取得官方数值flux和效率表，用它们替换解析Michel、均匀轴向几何和恒定效率。
3. 若只有Fig. 24的 $L/E$ 投影，可建立带明确图像读取误差的shape-only近似；它仍不是四维官方likelihood，且不能与同事件总率项无协方差地直接相加。
4. 更精确的IBD截面只能改善较小的能量权重误差，不能补回缺失的事件级形状和背景PDF。

当前与LSND官方的主要差距是 **事件likelihood维度和背景/响应信息**，不是振荡概率内核。当前与Kopp等 $1+3+1$ 全球拟合的主要差距是 **缺少其11-dof LSND likelihood输入**。仅升级flux曲线或IBD高阶修正不能跨越这两个层级。

### 5.12 建议插图

本仓库已有：

```text
outputs/lsnd_final_2001/three_plus_one/public_rate_approximation_20260906_v2/parameter_space.png
outputs/lsnd_final_2001/three_plus_one/public_rate_approximation_20260906_v2/parameter_space_lines.png
```

官方论文建议查看：Fig. 3（DAR/DIF flux形状）、Fig. 24（$L/E$投影）、Fig. 27（最终允许区）、Fig. 28（四维fit投影）以及Sec. II和Sec. IX。Kopp等全球拟合建议查看Table 1、Sec. 5、Sec. 6.2及其技术附录。

---

## 6. 三个实验的统计方法对照

| 实验/模式 | 本地检验量或likelihood | 本地profile | 本地Toy | 官方对照 |
|---|---|---|---|---|
| MicroBooNE `analytic` | $T=\chi^2_4-\chi^2_3$；广义二次型分布反演后计算 $CL_s$ | Fig.3a profile $s_{14}$；Fig.3b在两支 $s_{14}$ 上profile $s_{24}$ | 否 | 官方使用伪实验 $CL_s$ |
| MicroBooNE `toy` | 同一 $T$，经验右尾计算 $CL_s$ | 只对观测数据每点profile一次 | 是；每个假设指定 $N_{\rm toy}$ 份，Toy内不重新profile | 是否逐Toy重新profile未由当前公开文字充分确认 |
| MicroBooNE `adaptive-toy` | 候选带用Toy，其余点用广义二次型 | 同上 | 只在候选带 | 官方不是这种人为混合输出 |
| MicroBooNE $1+3+1$ `analytic` | 固定profile点的广义二次型 $CL_s$ | 固定质量对，profile四个模平方和一个相位 | 否 | 不是官方发布分析 |
| MicroBooNE $1+3+1$ `toy` | 固定profile点的经验Toy $CL_s$ | 观测数据profile一次 | 是；Toy内不重新profile | 不是官方发布分析 |
| MiniBooNE `scan` | 参数相关协方差的Gaussian NLL | 无额外profile | 否 | 与官方likelihood面和官方频率学派轮廓比较 |
| MiniBooNE `official` | 直接读取官方likelihood面 | 官方面已包含其原分析选择 | 本地不生成 | 官方轮廓为发布产品 |
| LSND `rate-scan` | 一维平均概率Gaussian NLL | 无 | 否 | 官方为DAR+DIF四变量事件likelihood的恒定切片 |

---

## 7. 哪些内容能统一，哪些必须保持实验特异

### 7.1 已经或应当完全统一

- $3+1$ 与 $1+3+1$ 真空振荡概率核心；
- 参数单位和质量平方差定义；
- profile调度和优化器边界记录；
- $CL_s$、广义二次型分布和Toy生成器；
- CSV/JSON写入；
- 参数空间图和谱图的公共渲染器；
- 数据来源、近似和随机种子的metadata记录；
- 多实验独立likelihood的求和接口。

### 7.2 必须由每个实验adapter处理

- 原始文件格式和bin顺序；
- 初始味道、末态选择和背景分类；
- flux、截面、效率和迁移在公开数据中以什么形式出现；
- kernel如何构造；
- covariance是绝对还是分数、是否依赖预测；
- 控制样本和信号样本的交叉相关；
- 哪些成分能随振荡概率重加权；
- 官方likelihood是Gaussian、Poisson、事件级PDF还是已发布表面；
- 官方置信区是否经过Toy/FC/$CL_s$校准。

统一接口的目标不是把所有实验伪装成相同的Reco矩阵，而是让不同adapter最终提供可审计的

$$
-2\ln L_i(\boldsymbol\eta)
$$

以及必要的预测、观测和metadata。

---

## 8. 当前可以运行的入口

### MicroBooNE输入与谱

```powershell
python run.py check --analysis all
python run.py spectrum --kind bnb --compare-paper-figure1-points
python run.py spectrum --kind joint
python run.py spectrum --kind figure1
```

### MicroBooNE参数扫描

```powershell
# Fig. 3a：广义二次型分布反演，无Toy
python run.py scan --preset fig3a --calibration analytic

# Fig. 3b：广义二次型分布反演，无Toy
python run.py scan --preset fig3b --calibration analytic

# 全网格固定profile点Toy；默认示例每个假设100份
python run.py scan --preset fig3a --calibration toy --number-of-toys 100

# 只在解析CLs候选带使用Toy
python run.py scan --preset fig3a --calibration adaptive-toy --number-of-toys 100
```

### 当前 $1+3+1$ MicroBooNE质量对开发扫描

```powershell
python run.py scan --model 1+3+1 --preset mass-pair --calibration analytic
```

### MiniBooNE

```powershell
# 读取并绘制官方发布面
python run.py miniboone --kind official

# 用公共3+1核心和官方逐事件输入重建本地NLL
python run.py miniboone --kind scan
```

### LSND

```powershell
# 导出论文明确打印的标量
python run.py lsnd --kind official

# 检查MeV/m单位和appearance振幅映射
python run.py lsnd --kind core-mapping

# 公开DAR总率近似likelihood
python run.py lsnd --kind rate-scan
```

---

## 9. 最终科学定位

当前仓库不是三个实验都达到同一严格程度的全局拟合程序，而是已经建立了统一的模型和实验adapter边界：

1. MicroBooNE 提供最完整的多通道重加权和 $CL_s$ 框架，但 detector kernel、遗漏通道、固定背景和 NuMI 输入仍有公开信息限制；
2. MiniBooNE 提供最可靠的逐事件 $3+1$ appearance 内核验证，但尚未成为完整的 $1+3+1$ appearance+disappearance实验项；
3. LSND 目前只提供最保守的DAR总率 likelihood，用于验证单位、平均概率和未来模型接口，不能复现官方四变量结果；
4. 当前只有 MicroBooNE adapter 接入了 $1+3+1$ mass-pair profile；MiniBooNE和LSND尚未纳入统一数值全局项；
5. 在补足每个实验的模型可移植输入前，任何多实验 $1+3+1$ 结果都应称为“在声明近似下的条件性联合分析”，不能称为合作组级严格全球限制。
