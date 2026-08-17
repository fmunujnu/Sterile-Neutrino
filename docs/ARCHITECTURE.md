# 架构与科学边界

## 当前科学范围

活动分析只使用 MicroBooNE BNB 的前四个公开通道：`nue_cc_fc`、`nue_cc_pc`、`numu_cc_fc`、`numu_cc_pc`，各 26 个 bin。NuMI 和 BNB 的三个 pi0/NC 通道不进入当前 likelihood；没有相应参数相关模板时，不能把它们假装为严格模型输入。

参考预测是 HEPData 的公开 `Signal + Background`；公开 `Background` 冻结并只加一次。`build_anchor.py` 采用声明的 per-reco-bin reference-ratio 方法构造 60-bin true-energy kernel，并要求在参考点逐 bin 精确复现公开总预测，否则 profile 运行被拒绝。

公开的 364×364 文件是系统协方差，已在 `data/experiments/microboone/shared/raw/` 中；BNB 私有的 `scripts/experiments/microboone/bnb/prepare_bnb_total_covariance.py` 会以 Pearson 统计项在公开参考谱处合成并保存 104×104 参考总协方差 CSV。实际扫描由 `PredictionScaledGaussianLikelihood` 在每一点用 `P/P0` 外积缩放系统协方差，并加入 `diag(P)`。HEPData 表头的 CNP 表述不能静默替换论文 Methods 的 Pearson 处方。

## 模块边界

- `parameters.py`：唯一允许参数格式转换的地方。外部配置从不使用含糊的 `dm41` 或 `theta14`。
- `models/three_plus_one.py`：只计算振荡概率与混合矩阵，不读取任何数据。当前 3+1 实现采用论文短基线极限：前三个质量态严格退化，只保留 `delta_m2_41_eV2`。
- `experiments/microboone/bnb/published_inputs.py`：只读取、验证和切片 BNB 公开数据。
- `covariance.py`：只构造和求解协方差。统计处理需显式声明。
- `experiments/microboone/bnb/templates.py`：定义 BNB 必须提供的 detector-folded 真能量模板。它严格区分
  `fixed_published_background_counts` 与 `beam_*_response_counts`；后者
  是随振荡概率重加权的束流中微子 CC 成分；前者在当前经验近似中明确等于
  HEPData `Background` 类别，但这个固定选择不等于宣称其中所有物理成分都不振荡。
- `experiments/microboone/bnb/prediction.py`：强制 BNB 模板在参考点闭合；不做经验校正。
- `likelihood.py`、`fitting.py`：统计推断；不包含物理常数或文件路径。
  `fitting.py` 还提供固定精确 `sin²(2θμe)` 后，在物理约束曲线上 profile
  `sin²θ14`、`sin²θ24` 的二维扫描，搜索完整两个 `sin²θ14` 分支，并显式检查零外观边界；禁止用 `θ14=θ24` 代替 profile。

BNB 的单基线近似值 `0.4685 km` 只在 `configs/experiments/microboone/bnb/analysis.yaml` 声明，并同时进入 anchor 构造和扫描预测；旧代码中的 `0.541 km` 属于错误实验基线，不得恢复。

## 无法由现有公开输入消除的限制

HEPData 只给出聚合 `Background`，没有给出其中每个可振荡中微子成分的真能量与来源模板。因此活动预测只能冻结整个公开背景块；这能保证参考谱记账正确，但不能严格复制合作组对可振荡背景的逐成分更新。同理，公开系统协方差没有拆出论文所述保持不变的非中微子和探测器外成分，当前整谱分数缩放是明确记录的近似。只有获得这些分量模板后，才能声称严格论文级 BNB likelihood。

## 未来 1+3+1 和全局拟合

`models/base.py` 定义模型接口。1+3+1 必须实现相同的概率接口，并在第二个惰性态混合为零时回归 3+1。其他实验通过独立 likelihood provider 加入；不得共享或假定 MicroBooNE 的探测器核。

每个实验和束流在 `experiments/<detector>/<beam>/` 中拥有自己的读取、适配、kernel、预测和
单实验谱图。`analysis/selection.py` 读取可见配置，`analysis/registry.py` 构建被选中的私有
workflow，`analysis/combination.py` 只接收具名标量 χ²，最终 `scripts/scan.py` 不读取任何
实验文件格式。

可直接相加的贡献必须来自不同 `correlation_group`。若 BNB 与 NuMI 使用跨束流协方差，
两条单束流 χ² 不得直接求和；必须由一个联合 workflow 读取完整块预测和块协方差，再作为
单个贡献交给组合层。这使单束流分析、独立实验组合和相关数据联合拟合三种情况都有明确边界。

冻结目录不在活动导入路径内。活动分析只读取从历史束流数组一次性语法转换得到的可见 `data/experiments/microboone/bnb/inputs/bnb_flux.csv`，并在 JSON 中记录原文件哈希；不会导入或执行冻结代码。
