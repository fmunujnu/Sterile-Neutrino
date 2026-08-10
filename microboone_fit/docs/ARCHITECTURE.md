# 架构与科学边界

## 当前科学范围

活动分析只使用 MicroBooNE BNB 的前四个公开通道：`nue_cc_fc`、`nue_cc_pc`、`numu_cc_fc`、`numu_cc_pc`，各 26 个 bin。NuMI 和 BNB 的三个 pi0/NC 通道不进入当前 likelihood；没有相应参数相关模板时，不能把它们假装为严格模型输入。

参考预测是 HEPData 的公开 `Signal + Background`。活动预测器不施加逐 bin 校正：外部提供的真能量、按味道过程拆分模板必须自行在参考点复现公开预测，否则 profile 运行被拒绝。

公开的 364×364 文件是系统协方差，已在 `data/raw/` 中；`scripts/prepare_bnb_total_covariance.py` 会以论文方法部分的 Pearson 统计项在该公开参考谱处合成 104×104 总协方差，并将参考谱哈希写入归档。HEPData 表头的 CNP 表述与论文不一致，不能在论文复现中静默替换统计处方。

## 模块边界

- `parameters.py`：唯一允许参数格式转换的地方。外部配置从不使用含糊的 `dm41` 或 `theta14`。
- `models/three_plus_one.py`：只计算振荡概率与混合矩阵，不读取任何数据。
- `published_inputs.py`：只读取、验证和切片公开数据。
- `covariance.py`：只构造和求解协方差。统计处理需显式声明。
- `templates.py`：定义必须提供的 detector-folded 真能量模板。它严格区分
  `fixed_non_oscillatory_background_counts` 与 `beam_*_response_counts`；后者
  是随振荡概率重加权的束流中微子 CC 成分，前者绝不能直接等同于公开表中
  笼统命名的 `Background`。
- `prediction.py`：强制模板在参考点闭合；不做经验校正。
- `likelihood.py`、`fitting.py`：统计推断；不包含物理常数或文件路径。
  `fitting.py` 还提供固定精确 `sin²(2θμe)` 后，在物理约束曲线上 profile
  `sin²θ14`、`sin²θ24` 的二维扫描，禁止用 `θ14=θ24` 代替 profile。

## 未来 1+3+1 和全局拟合

`models/base.py` 定义模型接口。1+3+1 必须实现相同的概率接口，并在第二个惰性态混合为零时回归 3+1。其他实验通过独立 likelihood provider 加入；不得共享或假定 MicroBooNE 的探测器核。

冻结目录不在活动导入路径内。它只用于历史对照，绝不用于预测、拟合或 profile。
