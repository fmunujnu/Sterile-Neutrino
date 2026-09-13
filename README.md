# 惰性中微子 profile 分析

当前代码按四块组织：实验输入、适配与调度、核心计算、统一输出。
日常只使用根目录 run.py；不再维护 run1/run2 和两个模型的多套脚本入口。
所有实验特异代码和命令适配器均位于 `src/sterile_fit/experiments/<experiment>/`；
包顶层不再保留 MiniBooNE、LSND 等实验同名入口文件。

- 长期原则：[AGENT.md](AGENT.md)
- 当前目录、调用和参数位置：[架构说明](docs/ARCHITECTURE.md)
- 每次修改要更新什么：[维护清单](docs/MAINTENANCE.md)
- 本次迁移的验证范围：[验证记录](docs/VALIDATION.md)
- 三个实验逐项输入、计算、profile、Toy与官方差异：[实验实现说明](docs/EXPERIMENT_IMPLEMENTATION_NOTE.md)

## 运行环境

Python >= 3.11。依赖声明在 pyproject.toml。从仓库根目录运行。
已有环境无需重新安装依赖；新环境可使用 python -m pip install -e ".[dev]"。
模型、统计和优化设置不会因为选择新的入口而自动改变。

在 `relics2` 上使用固定入口，无需填写仓库或数据的绝对路径：

```bash
cd ~/data/jobs/sterile-neutrino
bash scripts/server/run_relics2.sh check
bash scripts/server/run_relics2.sh scan --preset fig3a --calibration toy --number-of-toys 100
```

服务器工作副本使用 Git sparse-checkout，只展开 `src/`、`configs/`、`data/`
以及根目录入口。Windows 端提交后运行 `scripts/server/sync_relics2.ps1` 即可同时
推送 GitHub、服务器私有 Git 仓库并更新工作副本。脚本拒绝同步未提交改动。
服务器 Python 环境默认位于 `~/data/venvs/sterile-py311`；如管理员调整位置，
只需设置一次 `STERILE_VENV`，不修改代码。正式计算的软件版本记录在
`requirements-server.lock`。

## 检查与谱图

```powershell
python run.py check
python run.py spectrum --kind bnb --compare-paper-figure1-points
python run.py spectrum --kind joint
python run.py spectrum --kind figure1
```

figure1 是上 BNB、下 NuMI 的 nue CC FC 两面板，三个谱入口调用同一渲染函数。
它们画固定公开预测和指定参数点，不再偷偷执行全局 fit。
输入图使用 python run.py inputs --kind public 或 --kind numi-flux。

MiniBooNE 2020 ν+反ν联合发布采用平行入口，不改变MicroBooNE：

```powershell
# 原样绘制合作组发布的似然面和频率学派轮廓
python run.py miniboone --kind official

# 从发布的逐事件信号、背景、控制样本和协方差重算Gaussian NLL
python run.py miniboone --kind scan
```

`official` 是官方数值的直接可视化；`scan` 使用本仓库3+1短基线核心的精确
appearance振幅重建。对MiniBooNE公开的两味appearance模型它与原公式等价；只有通过逐面比较后
才能称为统计复现。二者不会被混作同一结果。
`scan` 另外输出无热力图的 `parameter_space_line_overlay.png`：实线是本地二维
似然固定阈值，点线是合作组发布的频率学派覆盖率轮廓，仅用于诊断二者差异。

LSND final 2001 uses the same parallel-but-not-joint-validation layout:

```powershell
# Export the collaboration paper's explicitly transcribed scalar record.
python run.py lsnd --kind official

# Audit only the LSND MeV/m convention and 3+1 appearance-amplitude mapping.
python run.py lsnd --kind core-mapping

# Public-input DAR rate-only 3+1 likelihood scan (explicit approximation)
python run.py lsnd --kind rate-scan
```

Unlike MiniBooNE, LSND did not release an event table, four-variable PDFs,
background-variation inputs, numerical likelihood surface, or numerical
contours. `official` and `core-mapping` therefore make no likelihood claim;
`rate-scan` produces a separately labelled public-input DAR total-rate
approximation, not the collaboration four-variable likelihood or an official
coverage contour. See `data/experiments/lsnd/`.

## Profile：选择近似还是 Toy

```powershell
# 原 Fig3a 联合分析坐标和范围，Gaussian分布近似
python run.py scan --preset fig3a --calibration analytic

# Fig3b联合分析；质量平方差范围0.1--40 eV^2
python run.py scan --preset fig3b --calibration analytic

# Toy：每个假设每点 100 份，保守使用单进程
python run.py scan --preset fig3a --calibration toy --number-of-toys 100 --scan-workers 2 --toy-workers 1

# 解析全图 + 指定带内 Toy；不是全 Toy 图
python run.py scan --preset fig3a --calibration adaptive-toy --adaptive-analytic-cls-min 0.01 --adaptive-analytic-cls-max 0.3 --number-of-toys 100 --scan-workers 1 --toy-workers 1

# 并行开发模型；保持原 7x7 默认质量网格和 profile 设置
python run.py scan --model 1+3+1 --preset mass-pair --calibration analytic
```

3+1 扫描默认在 `outputs/.scan_cache/three_plus_one/` 保存内容寻址的观测数据profile和二次型前置缓存。相同活动代码、配置、科学输入、网格和profile模式下，仅改变Toy数、种子、批大小或自适应Toy范围会直接复用此前置阶段；输出metadata明确记录是否命中。每个扫描点只profile观测数据一次，随后缓存并固定该点的3nu/4nu预测、协方差及Cholesky分解；同一批Toy使用精确的批量二次型求解，不在Toy内部重新profile。`--no-precalibration-cache` 可强制完整重算。缓存是可删除的派生产物，不是科学输入。

`--scan-workers N` 同时用于彼此独立且保持原顺序的3+1 profile点和二次型点；数值算法、边界与容差不变。本机仍应从较小的N开始，避免底层线性代数线程叠加。

完整 61x61 Toy 扫描开销很大；100 Toy/假设只适合初步诊断，不是精确 0.05 尾部。
只检查程序时使用较小显式网格或 --grid-points 8，不要默认启动完整 Toy。
不指定 preset 时，3+1 保留原扫描器的 BNB-only、61x61 默认设置。
fig3a/fig3b preset 显式选择近似 BNB+NuMI；mass-pair 默认 BNB-only。

```powershell
python run.py scan --engine-help
python run.py scan --model 1+3+1 --engine-help
```

完整选项见上述帮助；对模型扫描器显示的 --cls-calibration，在统一入口写 --calibration。
--mode prefit 和 --compare-fit-points 已归档并拒绝使用。
profile 仍然执行原来的内部最小化；归档的是独立全局 prefit，不是取消最小化。

## 结果

统一规则：`outputs/来源/模型/批次/产物/`。详细目录和历史迁移说明见 [输出说明](docs/OUTPUTS.md)。

```text
outputs/microboone_bnb_numi_joint/three_plus_one/my_run/
  spectra_figure1/          谱图、逐bin CSV、原metadata
  scan_fig3a_analytic/      result.csv、profile.png、metadata.json
  scan_fig3b_analytic/
  scan_fig3a_adaptive-toy/  同类结果，方法明确区分
  contour_comparison/      读取指定结果重绘的对比图
  provenance/              每次调用的参数和代码/配置SHA256
```

谱、扫描、输入图、准备和比较入口支持 `--batch my_run`；不指定时用独立UTC时间戳，避免不同运行混放。
相同来源/模型下使用同一批次名，可把不同产物集中起来；同产物再跑请用新批次，避免覆盖（3+1扫描已存在则拒绝）。
`--output`、`--output-directory` 仍可显式指定位置，优先于自动布局；此时不生成批次provenance，原产物metadata照常生成。
科学输入及kernel仍存data，准备命令不会把它们改存outputs。

扫描输出仍有 result.csv、metadata.json 和 profile.png。
chi2、3nu chi2、T、profile 参数及 CLs 保留；独立 prefit 对应的全局 best_fit_found、
prefit_seeds、delta_chi2 诊断不再生成。CLs 定义、概率近似和 Toy 算法没有因此改变。
用 python run.py compare 显式指定四个结果目录叠加轮廓，不重新拟合或校准；完整命令见输出说明。

## 输入准备：只在明确需要重建时运行

```powershell
python run.py prepare --kind reco-normalize
python run.py prepare --kind reco-bnb26
python run.py prepare --kind bnb-covariance
python run.py prepare --kind bnb-kernel
python run.py prepare --kind numi-flux
python run.py prepare --kind numi-kernel
```

这些命令会写入 data 中的可复用整理输入；不是日常扫描的前置自动步骤。
本次结构迁移没有执行它们，没有重建或修改科学输入。

## 当前科学范围

- BNB 四通道、104 bins：公开预测经验锚定。
- BNB+NuMI：208 bins，保留发布的跨束流协方差；NuMI正式读取登记的公开dk2nu条件基线分布并作能量、味道相关平均，但仍使用借用的BNB Reco先验，因此是活动近似分析，不是合作组复现。
- Fig3b现扫描0.1--40 eV^2；当前psi以10 m基线bin中心评价相位，高质量差区的bin内快速振荡尚未做收敛验证，不能把扩展显示范围等同于新增可靠灵敏度。
- 单独 NuMI 有输入/预测模块，但没有独立注册的扫描选择。
- 固定公开 Background、单基线、未知截面/效率由经验 kernel 吸收等限制仍然存在。
- HEPData 总谱作为零混合锚点的当前声明未在结构迁移中重新裁决。
- 1+3+1 质量对平面允许全部混合归零，所以只是当前开发诊断，不可直接宣称整个模型被排除。
- analytic 对观测数据逐点profile后，以固定假设下解析得到的T均值和方差作Gaussian近似；toy固定相同逐点假设并用批量矩阵求解获得经验分布；adaptive 是Gaussian预选与Toy的显式混合。广义二次型特征函数反演不再进入活动扫描。
- 不是合作组完整 14 通道内部分析。
- MiniBooNE 当前是独立的两味 appearance 验证入口，尚未加入跨实验联合fit；官方
  轮廓做过频率学派覆盖率研究，本地 `scan` 目前只重建Gaussian NLL，不冒充该校准。
  当前3+1接入只计算发布包可识别的appearance振幅；没有足够公开事件分类来对所有
  背景和muon控制样本实施完整的3+1 disappearance重加权。
- LSND final 2001 is an appearance-only public-fact and 3+1-unit-convention
  audit. The final event likelihood and numerical surface were not publicly
  released, so it is not a joint-fit input and is not a likelihood reproduction.

本文随入口、默认设置、输出或科学范围变化更新；不存放长期不变原则。
