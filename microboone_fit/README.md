# MicroBooNE BNB 3+1 profile analysis

这是一个为 **MicroBooNE BNB 前四个通道、104 个重建能量 bin** 建立的 3+1 剖面似然代码库。它的首要规则是：参数、数据顺序、模板物理含义和统计处理必须显式可检验。

当前未实现 1+3+1；未来模型只能通过 `models/base.py` 的概率接口接入，并先证明在相应极限回到 3+1。

## 当前能严格做什么

1. 读取并验证公开 14×26 数据与 364×364 协方差。
2. 明确选择 BNB 前四个 CC 通道的 104-bin 子空间。
3. 用无歧义的 3+1 参数计算真空振荡概率。
4. 在提供**真能量、按味道过程拆分的探测器折叠模板**及**完整总协方差**后，执行严格定义的 profile likelihood。
5. 从公开 HEPData 数字表重画 BNB 四通道的 data / background / signal+background 面板。

它不能仅凭公开最终重建谱自动恢复合作组的内部 MC、效率或截面；缺失模板时 profile 脚本会拒绝运行，而不会悄悄退回旧代码或猜测输入。

## 安装与基础验证

```powershell
cd "E:\Sterile Neutrino\microboone_fit"
python -m pip install -e ".[dev]"
python -m pytest -q
python -m sterile_fit.cli inspect-published
python -m sterile_fit.cli validate-3plus1
```

如果未安装为 editable package，可临时设置：

```powershell
$env:PYTHONPATH = "src"
```

## 重画公开 BNB 图片

```powershell
$env:PYTHONPATH = "src"
python scripts/plot_published_bnb.py --output runs/reproduction/published_bnb_four_channels.png
```

该命令严格使用公开数表的 data、background、signal+background 和统计误差，输出四个 BNB 通道的可复核面板。它复现的是公开数值输入；不会声称复现论文完整 BNB+NuMI 图形、合作组作图样式或内部拟合结果。

## 严格 profile 所需的两个输入

在运行 profile 前，必须提供有来源的外部 MC 模板；总协方差可由本仓库已有的公开系统协方差加公开数据表的 CNP 统计项构造。

### `data/derived/bnb_four_channel_oscillation_templates.npz`

必须包含：

```text
true_energy_GeV                 (n_true,)
    fixed_non_oscillatory_background_counts        (104,)
    beam_nue_to_nue_cc_response_counts             (104, n_true)
    beam_numu_to_nue_cc_response_counts            (104, n_true)
    beam_nue_to_numu_cc_response_counts            (104, n_true)
    beam_numu_to_numu_cc_response_counts           (104, n_true)
    beam_nuebar_to_nuebar_cc_response_counts       (104, n_true)
    beam_numubar_to_nuebar_cc_response_counts      (104, n_true)
    beam_nuebar_to_numubar_cc_response_counts      (104, n_true)
    beam_numubar_to_numubar_cc_response_counts     (104, n_true)
```

每个 `beam_*_response_counts` 元素是在真能量 bin 中、相应束流初始味道到末态 CC 过程的探测器折叠事件数，已经包括 BNB 通量、截面、固定效率、选择和 Reco 迁移。中微子与反中微子模板必须分开提供；程序只再乘相应的振荡概率，绝不重复乘效率或截面。

`fixed_non_oscillatory_background_counts` 只能容纳已明确证明不随这套振荡概率变化的成分，例如宇宙线或仪器背景。公开表中笼统名为 `Background` 的向量不能直接填入此处：它须先与本征束流 νe、νμ CC 成分逐项核对，避免重复计数。

### `data/derived/bnb_four_channel_total_covariance.npz`

必须包含：

```text
covariance                       (104, 104)
statistical_treatment            scalar string
parameter_dependence             "fixed_at_reference"
reference_prediction_sha256      scalar string
provenance                       scalar string
```

仓库已含公开的 364×364 **系统**协方差；其表头明确排除数据统计项。目标 2025 论文的方法部分使用 Pearson 统计协方差，因此默认命令以公开 `Signal + Background` 参考谱构造 BNB 前四通道的固定参考 Pearson 总协方差：

```powershell
$env:PYTHONPATH = "src"
python scripts/prepare_bnb_total_covariance.py --statistical-method pearson
```

生成文件会记录参考谱 SHA-256 和来源；profile workflow 会验证它正对应当前公开参考谱。HEPData 表头和论文在 CNP/Pearson 描述上不一致，故 `--statistical-method cnp` 只用于明确的交叉检查，不能默认称为论文复现。当前 likelihood 仅支持 `parameter_dependence="fixed_at_reference"`；若论文目标要求每个参数点重新构造统计协方差，必须先实现专门的 `C(parameters)` likelihood。

模板在参考点必须逐 bin 复现 HEPData 的 `Signal + Background`；否则脚本立即停止。

## 运行预拟合与 profile

以下参考点只是示例参数，不自动称为论文 best fit：

```powershell
$env:PYTHONPATH = "src"

python scripts/run_prefit.py `
  --templates data/derived/bnb_four_channel_oscillation_templates.npz `
  --total-covariance data/derived/bnb_four_channel_total_covariance.npz `
  --reference-delta-m2-eV2 1.2 `
  --reference-sin2-theta14 0.041666666666666664 `
  --reference-sin2-theta24 0.018

python scripts/run_profile.py `
  --templates data/derived/bnb_four_channel_oscillation_templates.npz `
  --total-covariance data/derived/bnb_four_channel_total_covariance.npz `
  --reference-delta-m2-eV2 1.2 `
  --reference-sin2-theta14 0.041666666666666664 `
  --reference-sin2-theta24 0.018 `
  --delta-m2-grid-eV2 0.1,0.3,1.0,3.0,10.0 `
  --sin2-theta14-grid 0.001,0.01,0.05
```

第二条命令在每个固定的 `(Δm²41, sin²θ14)` 点，对未固定的 `sin²θ24` 做全局有界最小化；这才是该二维扫描的 profile likelihood。它严格对应于“模板和总协方差均已声明、协方差固定在参考点”的统计模型。

若目标是论文使用的 `sin²(2θμe)–Δm²41` 平面，应使用精确约束而不是小角近似或强制 `θ14=θ24`：

```powershell
python scripts/run_profile.py `
  --templates data/derived/bnb_four_channel_oscillation_templates.npz `
  --total-covariance data/derived/bnb_four_channel_total_covariance.npz `
  --reference-delta-m2-eV2 1.2 `
  --reference-sin2-theta14 0.041666666666666664 `
  --reference-sin2-theta24 0.018 `
  --delta-m2-grid-eV2 0.1,0.3,1.0,3.0,10.0 `
  --sin2-2theta-mue-grid 0.0001,0.001,0.01,0.1
```

每个点固定 `sin²(2θμe)=4 sin²θ14 (1-sin²θ14) sin²θ24`，在 `0≤sin²θ14,sin²θ24≤1` 的物理域内 profile 剩余自由度；不会把 `θ24=0` 或 `θ14=θ24` 当成默认假设。

## 为什么不写成一个大文件

大文件会把四种不同的错误混在一起：参数单位错误、振荡公式错误、bin 顺序错误和 χ²/优化错误。这里把它们拆开：

- `parameters.py`：物理参数和单位；
- `models/three_plus_one.py`：仅振荡概率；
- `templates.py`：探测器折叠模板的唯一物理定义；
- `published_inputs.py`：公开数据与 bin 映射；
- `covariance.py`：统计协方差；
- `prediction.py`：模板对公开标称预测的严格闭合；
- `fitting.py`：全局预拟合和条件 profile；
- `workflows.py`：把已验证的输入组合为可运行分析；
- `tests/`：每个边界的失败测试。

这使未来 1+3+1 或其他实验的加入不需要重写、复制或污染已验证的 BNB 3+1 核心。

完整规则见 [AGENT.md](AGENT.md) 和 [架构说明](docs/ARCHITECTURE.md)。
