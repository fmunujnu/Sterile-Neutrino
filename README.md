# MicroBooNE BNB reference-reweight analysis

本项目使用 MicroBooNE 公开 BNB 前四个 CC 通道，在一个明确声明的 3+1 参考点上构造可审计的经验重加权核。它不寻找或声称恢复合作组原始 MC；所有活跃数值输入均为 CSV/YAML，所有元数据与结果均为 JSON/CSV。

## 最常用的三个命令

首次安装：

```powershell
cd "E:\Sterile Neutrino"
python -m pip install -e ".[dev]"
```

完整检查：

```powershell
python scripts/check.py
```

重画 HEPData 公开 BNB 四通道图：

```powershell
python scripts/experiments/microboone/bnb/plot_published_bnb.py
```

如需同时检查经验锚点闭合和本项目 BNB-only best fit：

```powershell
python scripts/experiments/microboone/bnb/plot_published_bnb.py --compare-fit-points --output outputs/spectra/microboone/bnb/published_four_channels_fit_comparison.png
```

黑线是 HEPData `Unconstrained Signal + Background`。紫线把该谱强制指定为零混合经验锚点，
所以逐 bin 重合只是代数闭合，不是物理验证；官方发布没有把这张 unconstrained 表明确
定义为论文的 3ν/null prediction，因此该锚点不得用于声称严格复现论文零假设。本图
只画本地 4 通道 BNB-only 经验重加权 best fit，不混入任何 NuMI 联合拟合参数。

用本地 BNB kernel 绘制论文 Figure 1 的两个示意参数点：

```powershell
python scripts/experiments/microboone/bnb/plot_published_bnb.py --compare-paper-figure1-points --output outputs/spectra/microboone/bnb/paper_figure1_parameter_comparison.png
```

两个点共同使用 `delta_m2_41_eV2=1.2`、`sin2_2theta_mue=0.003`，并分别使用
`sin2_theta24=0.018` 和 `0.0045`。只有 `nue_cc_fc` 面板与论文 Figure 1 的 BNB
面板直接对应；曲线由本项目经验 kernel 计算，不冒充合作组原始 MC。

运行精确 `sin²(2θμe)-Δm²41` profile：

```powershell
python scripts/scan.py --mode appearance-profile
```

运行包含发布 BNB--NuMI 交叉协方差的 208-bin 联合 prefit：

```powershell
python scripts/scan.py --mode prefit --analysis-config configs/analyses/microboone_bnb_numi.yaml
```

联合谱图与逐 bin 表：

```powershell
python scripts/experiments/microboone/plot_bnb_numi_joint.py
```

联合 workflow 在组合层表现为一个具名 χ²；内部一次性读取完整块协方差，不能拆成
BNB χ² 与 NuMI χ² 后相加。其他独立实验仍按原有 `ChiSquareContribution` 接口加入。

扫描结果写入 `outputs/scans/<分析名>/three_plus_one/<时间>_appearance-profile/result.csv`，运行说明写入同目录的 `metadata.json`。
默认使用两个方向各 61 点的对数网格（共 3721 个 profile 点）。可用 `--grid-points 81` 做更高分辨率收敛检查，或用 `--grid-points 41` 较快测试；显式 grid 参数会覆盖默认坐标。少于 8×8 点时程序拒绝画容易误导的等高线。

## 可调扫描参数

```powershell
python scripts/scan.py `
  --mode appearance-profile `
  --delta-m2-grid-eV2 0.1,0.3,1.0,3.0,10.0 `
  --sin2-2theta-mue-grid 0.0001,0.001,0.01,0.1
```

其他模式：

- `--mode prefit`：同时拟合 `delta_m2_41_eV2`、`sin2_theta14`、`sin2_theta24`；
- `--mode s14-profile`：固定 `(delta_m2_41_eV2, sin2_theta14)`，profile `sin2_theta24`；
- `--mode appearance-profile`：固定 `(delta_m2_41_eV2, sin2_2theta_mue)`，在精确物理约束下 profile `sin2_theta14` 并推导 `sin2_theta24`。
- `--mode electron-disappearance-profile`：对应论文 Fig. 3b，固定
  `(delta_m2_41_eV2, sin2_2theta_ee)`；严格保留
  `sin2_theta14=(1±sqrt(1-sin2_2theta_ee))/2` 两个分支，并在每个分支 profile
  `sin2_theta24` 后选择较小 chi2。

一次生成联合 BNB/NuMI 谱以及论文坐标下的 Fig. 3a、Fig. 3b 诊断图：

```powershell
python scripts/generate_microboone_bnb_numi_figures.py --grid-points 61
```

该入口新增生成一个论文专用双面板谱图：上图为 BNB `nue_cc_fc`，下图为
NuMI `nue_cc_fc`；原有 BNB 四通道谱脚本及原有联合谱脚本保持原用途，不由此
入口替换。扫描范围固定为：Fig. 3a 的
`sin²(2θμe)=10^-4..1`、`Δm²41=10^-2..10^2 eV²`；Fig. 3b 的
`sin²(2θee)=10^-2..1`、`Δm²41=10^-1..14 eV²`。参数图只绘制红色
`CLs=0.05` 排除线，不绘制白色辅助 Δχ² 等高线。

谱图中的 BNB 和 NuMI 只提供不同的数据 payload，二者统一调用
`sterile_fit.spectrum_plotting.render_microboone_spectrum_panels`。Fig. 3a/3b
参数空间热图、颜色条和红色排除线均使用 `CLs`，最终排除判定为
`CLs=p_4nu/p_3nu<=0.05`。不带额外参数的扫描仍使用一、二阶矩高斯近似，作为
快速诊断。要求用伪实验直接测量两个检验量分布时，显式运行：

```powershell
python scripts/scan.py `
  --mode appearance-profile `
  --analysis-config configs/analyses/microboone_bnb_numi.yaml `
  --cls-calibration toy `
  --number-of-toys 100 `
  --toy-seed 20250821 `
  --toy-workers 4 `
  --toy-batch-size 256
```

Toy 模式在 3ν 与每个被检验的 4ν 点下各生成 `--number-of-toys` 份伪数据，使用
完整且随预测更新的协方差。每一份伪数据都重新执行与真实数据完全相同的二维点内
profile，再直接统计 `T=chi2_4nu-profile-chi2_3nu` 的右尾频率；不再把 T 假设为
高斯或卡方分布。Pearson 统计方差已经包含在协方差中，因此不会再叠加一次 Poisson
抽样，也不会裁掉高斯伪数据中的负值。

`result.csv` 保存两个尾部计数、经验 p 值、`CLs`、Monte Carlo 标准误以及 T 分布的
均值/标准差/5%、50%、95% 分位数。需要逐个检查全部 Toy 时增加
`--store-toy-distributions`，程序会为每个扫描点写一个可见 CSV。线程只并行彼此独立的
Toy profile；扫描点种子由 `--toy-seed` 和固定点序号确定，因此改变线程数不会改变随机
样本。`--toy-batch-size` 只限制同时驻留内存的伪数据数量，不改变随机流或结果。论文没有
公开 Toy 数、随机种子、被 profile 参数的伪数据生成方式和优化器细节；
本实现采用“真实数据在该扫描点的 profile 最优值”作为 4ν 生成点，并在 metadata 中明确
记录，不能把这些未公开选择声称为合作组原始设置。

当前默认是每个假设、每个扫描点 100 Toy。完整 61×61 网格因此包含 744200 次
带 profile 的伪实验拟合，适合生成初步轮廓和测量实际运行时间，但不足以稳定测量
`CLs=0.05` 附近的尾部。最终结果仍需增加 Toy 数、比较不同种子，并要求轮廓波动小于
所需线位精度。

优先推荐自适应模式：先在完整网格计算快速解析 CLs，再只对解析结果位于
`0.01..0.1` 的点及每个质量行左右各一个相邻点运行100 Toy：

```powershell
python scripts/scan.py `
  --mode appearance-profile `
  --analysis-config configs/analyses/microboone_bnb_numi.yaml `
  --cls-calibration adaptive-toy `
  --number-of-toys 100 `
  --toy-workers 4
```

`result.csv` 分别保存 `cls_asymptotic`、`cls_toy`、`adaptive_toy_candidate`、
`adaptive_toy_evaluated` 和用于初步绘图的 `cls_adaptive_hybrid`。未选择的点仍明确是
解析值，不得称为 Toy 结果。`--adaptive-toy-point-limit N` 只用于抽取横跨候选带的 N
个点做运行测试；正式自适应扫描必须省略该参数。

经验锚点与基线在 `configs/experiments/microboone/bnb/analysis.yaml` 中声明。该文件明确将“HEPData unconstrained
总谱＝零混合谱”标为不受论文支持的工作假设；它不能作为严格论文零假设复现。论文图 1
的示意点也只作为注释保留，绝不参与锚定。参数只使用：

```text
delta_m2_41_eV2    Δm²41，单位 eV²
sin2_theta14       sin²θ14
sin2_theta24       sin²θ24
```

BNB 基线也在同一配置中显式设为 `baseline_km: 0.4685`。这是论文给出的 BNB 靶到 MicroBooNE 探测器距离；当前 kernel 使用单基线近似，不冒充合作组逐事件产生点分布。

## 简化调用链

```text
scripts/check.py                                      一键检查当前被支持的输入与闭合
scripts/experiments/microboone/bnb/build_anchor.py    BNB 数据 → BNB 文本 kernel
scripts/experiments/microboone/bnb/plot_published_bnb.py 只画 BNB 单实验谱
scripts/scan.py                                       选择实验 → 合并 χ² → profile CSV
```

内部核心按职责保留：

```text
parameters.py                                      参数定义
models/three_plus_one.py                           精确 3+1 概率
covariance.py                                      通用协方差算法
likelihood.py                                      通用似然
fitting.py                                         通用 prefit 与 profile
analysis/selection.py                              显式选择纳入哪些实验
analysis/combination.py                            只负责相加各实验 χ²
experiments/microboone/bnb/                        BNB 私有读取、kernel、预测与 workflow
experiments/microboone/numi/                       NuMI 独立适配接口；likelihood 仍禁用
```

## HEPData 三个数据块

`HEPData-ins3088922-v1-Unconstrained_14_channels.csv` 中：

```text
Data                  只进入 χ²
Background            在参考重加权中冻结并只加一次
Signal + Background   参考总预测
Signal                 由 (Signal + Background) - Background 得到
```

`plot_published_bnb.py` 直接画 `Background` 和 `Signal + Background` 两条线，没有执行 `total + background`。

## 60 true bins 与 26 reco bins

不把 26-bin reco 谱插值成 60-bin true 谱，也不压缩 true-energy 轴。项目保留：

```text
60 true-energy bins：0--3 GeV，每 bin 0.05 GeV
26 reco bins：25 个 0.1 GeV bin + 2.5--3.0 GeV overflow
Reco 矩阵：26 x 60，R(reco|true)
```

`build_anchor.py` 在每个公开 reco bin 中，用 `Reco × 束流 × 参考概率` 分配 true-energy 权重，并用公开 Signal 强制归一化。因此参考点逐 bin 精确闭合，又不把 reco energy 冒充 true energy。

## 可见数据位置

```text
data/experiments/microboone/shared/         同一探测器跨束流共享的原始发布表
data/experiments/microboone/bnb/inputs/     BNB 束流等输入
data/experiments/microboone/bnb/derived/    BNB Reco 和协方差
data/experiments/microboone/bnb/reweighting/ BNB 重加权 kernel
data/experiments/microboone/numi/inputs/     NuMI 束流输入及来源
data/experiments/microboone/numi/derived/    NuMI 可复用适配数据与协方差
data/experiments/microboone/numi/reweighting/ NuMI 诊断 kernel 与参考闭合
```

活跃分析不读取 NPZ、pickle 或其他二进制数值容器。PNG 只作为可见图片输出，不作为拟合输入。

所有活跃生成物统一写入 `outputs/`：

```text
outputs/spectra/<detector>/<beam>/   单实验谱图及对应 CSV/metadata
outputs/scans/<analysis>/<model>/    prefit、profile 和参数空间结果
outputs/checks/                      人工或自动审计产物
outputs/testing/                     临时测试目录
```

`data/.../derived` 与 `data/.../reweighting` 是带来源信息的分析输入，不属于一次性输出。

## 科学边界

当前结果是“BNB 四通道、HEPData unconstrained 总谱经验锚定、公开背景类别冻结”的重加权分析，
不是严格的论文零假设复现。metadata 明确记录了以下假设：未知截面与效率不能从一个 reco
谱唯一反演，而是假定其参考加权效应可由 `Reco×flux` 真能量形状先验和每个 reco bin 的
一个比例因子吸收；同一选择通道内中微子和反中微子共享该有效比例；2022 Reco 作为
true-energy 分配先验。

配置注释中的 `1.2 eV², sin²θ24=0.018, sin²(2θμe)=0.003` 是论文图 1 的示意点，不是论文 best-fit，也不是 kernel 锚点。

这里的“固定背景”只表示 HEPData `Background` 类别在本近似中保持不变。公开表没有给出该类别内可振荡束流中微子、宇宙线、探测器外事件等逐成分真能量模板，因此这是当前最重要的不可消除限制。协方差在每个扫描点按当前/名义预测比保持分数系统误差，并重新加入当前预测的 Pearson 统计项；由于公开协方差也没有逐背景成分拆分，这仍是公开数据近似。论文正式限制还使用全部 14 通道和伪实验 `CLs`；本项目输出是 BNB-only 的诊断性 profile `Δχ²`，不能标成论文 95% 排除线。

profile 会搜索完整 `0 <= sin²θ14 <= 1` 与 `0 <= sin²θ24 <= 1` 域；固定外观振幅时不会丢弃大 `sin²θ14` 支。prefit 还显式检查 `sin²θ24=0` 和 `sin²θ14=0` 两个零外观边界，避免连续优化器漏掉边界极小值。

未来 1+3+1 只替换概率模型，必须在第二惰性态混合为零时逐点回到 3+1；数据、kernel、背景、Reco、协方差与 χ²不得复制或改变。

## 选择纳入哪些实验

最终扫描不再写死实验。编辑 `configs/analyses/microboone_bnb.yaml` 中每项的
`include` 即可选择进入总 χ² 的实验/束流。每项还必须声明状态：

```text
validated_surrogate   已验证但仍有公开数据近似
approximate           只允许作为明确标记的近似
inputs_unavailable    输入不完整，程序禁止启用
```

当前只注册了 `microboone.bnb.four_channel`。`microboone.numi` 明确禁用；将它的
`include` 改为 `true` 会立即失败，而不会静默使用 BNB 数据。未来每个实验先在自己的
目录中构造 `predict(parameters)` 和 `chi2(parameters)`，组合层只相加：

```text
global parameters → each experiment predictor/likelihood → named χ² terms → total χ² → common profile scan
```

`correlation_group` 防止把相关数据误当独立实验相加。BNB 与未来 NuMI 都属于
`microboone_2025_release`；若两者需要同时进入拟合，应注册一个读取完整跨束流块协方差的
联合 workflow。将两个同组的单束流 χ² 直接相加会被程序拒绝。
