# 仓库分类与运行边界

## 一、核心代码

`src/sterile_fit/` 是唯一可导入的核心包。这里按职责分为：

```text
parameters.py                 3+1参数及精确派生混合振幅
models/                       与实验无关的振荡概率
covariance.py                 协方差构造和线性代数求解
likelihood.py                 prediction-dependent Gaussian chi2
fitting.py                    prefit、二维profile和物理边界搜索
one_plus_three_plus_one/      平行1+3+1参数、概率、约束profile及现有实验适配
statistics/asymptotic_cls.py  Run 1使用的解析矩-高斯CLs近似
statistics/toy_cls.py         Run 2使用的经验Toy-MC CLs
analysis/                     实验选择、注册和独立贡献组合
experiments/                  各实验/束流的数据适配、kernel和预测
spectrum_plotting.py          BNB、NuMI共用的唯一谱图渲染器
```

核心包不读取 `outputs/`，不导入 `scripts/` 或 `studies/`，也不包含某一张
论文图片的临时逻辑。

## 二、两个正式运行类别

### Run 1：non-Toy

入口：

```powershell
python scripts/runs/run1_non_toy.py --mode appearance-profile
```

它强制使用 `--cls-calibration analytic`，不生成伪实验。内部根据

```text
T = chi2_4nu - chi2_3nu
```

在两个固定假设下的一、二阶矩作正态近似并计算两个右尾概率。结果只能称为
“non-Toy解析CLs诊断”，默认输出到 `outputs/run1_non_toy/`。

### Run 2：Toy MC

入口：

```powershell
python scripts/runs/run2_toy_mc.py --mode appearance-profile --number-of-toys 100
```

它强制使用 `--cls-calibration toy`。每个扫描点分别在3nu和被检验4nu假设下生成
完整协方差伪数据，并对每份伪数据重复相同profile。默认100 Toy/假设只用于初步
诊断；最终0.05尾部需要更多Toy和多随机种子稳定性检查。默认输出到
`outputs/run2_toy_mc/`。

两个入口都转交给 `scripts/scan.py`，因此不存在两份物理预测或两份profile代码。

1+3+1 是隔离的新增物理模型，入口为
`scripts/one_plus_three_plus_one/run1_non_toy.py` 与 `run2_toy_mc.py`。二者只共用
`scripts/one_plus_three_plus_one/scan.py`，调用既有实验kernel、协方差与CLs接口；
七参数profile不会写回或泛化现有3+1扫描器。

## 三、实验适配与可复现准备

```text
configs/analyses/                         选择纳入哪些实验
configs/experiments/                      基线、状态、路径和参考参数
data/experiments/<detector>/shared/raw/   不可修改的合作组公共发布
data/experiments/<detector>/<beam>/inputs 人可检查的束流输入
data/experiments/<detector>/<beam>/derived 可重复生成的Reco/协方差
data/experiments/<detector>/<beam>/reweighting 可见的重加权kernel
scripts/experiments/                      薄的准备与单实验谱图命令
```

BNB是当前严格活动接口。BNB+NuMI 208-bin联合分析保留完整交叉协方差，但NuMI仍
借用BNB响应，所以只能标记为近似诊断，不能称为合作组14通道严格复现。

## 四、研究性和一次性代码

所有不参与活动运行链的工具统一放在 `studies/`：

```text
studies/numi_flux_pdf_extraction/  从MicroBooNE Note矢量PDF提取NuMI flux的独立研究
studies/official_grid_profile/     对官方三维delta-chi2大网格做profile的比较研究
studies/scan_result_comparison/    对已经完成的Run 1/Run 2 CSV叠加画线
```

这些代码可以读取正式结果做检查，但不得成为 `src/`、正式数据或未来实验接口的依赖。

## 五、输出分类

```text
outputs/run1_non_toy/          non-Toy解析CLs扫描
outputs/run2_toy_mc/           全Toy或明确标记的Toy结果与轮廓比较
outputs/spectra/               各实验独立谱图及逐bin sidecar
outputs/checks/                输入、PDF提取、官方网格和性能检查证据
outputs/archive/legacy_runs/   整理前旧扫描，只保留追溯，不作为当前结果
```

旧的 `outputs/production/`、`outputs/paper_reproduction/` 和散乱
`outputs/scans/.../final_vN` 已停止使用。历史结果只读归档；新结果只允许进入上述稳定分类。

## 六、冻结历史

`frozen/baseline_v1/` 与 `frozen/current_system_backup/` 保留原始基准和完整旧体系，
不得导入、执行、修改或作为新代码兼容目标。它们与 `studies/` 不同：`studies/` 是可运行
但非核心的研究工具，`frozen/` 只是历史证据。

## 七、调用链

```text
run1_non_toy.py 或 run2_toy_mc.py
    -> scripts/scan.py
    -> analysis selection + registry
    -> BNB或联合实验workflow
    -> prediction + prediction-dependent covariance
    -> fitting/profile
    -> analytic_cls 或 toy_cls
    -> result.csv + metadata.json + profile.png
```

任何新增实验只增加自己的数据适配和workflow；不复制扫描、profile、CLs或绘图核心。
