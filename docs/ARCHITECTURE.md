# 当前架构与调用说明

此文描述会随代码变化的事实，不是长期规则。最后结构更新：2026-09-03。
长期规则见根目录 AGENT.md；同步责任见 MAINTENANCE.md。

## 四块

```text
run.py
src/sterile_fit/
  experiments/interface.py       所有实验面向公共分析层的likelihood/分箱预测接口
  experiments/microboone/
    adapter.py                  MicroBooNE选择、构建、模型接入和Toy假设适配
    public_data.py             通道定义、完整公开谱/协方差读取、两个束流选择
    response.py                原始Reco读取、零列/归一化、26-bin适配与准备
    bnb.py                     BNB kernel、预测、构建、闭合、谱数据组装
    numi.py                    NuMI kernel、预测、曝光加权、构建
    joint.py                   联合208-bin协方差、原缓存预测、联合/双面板谱组装
  experiments/miniboone/
    adapter.py                  MiniBooNE命令及官方/本地分析适配
    official_2020.py           官方ν+反ν输入、逐事件appearance预测、60到38维协方差
  experiments/lsnd/
    adapter.py                  LSND命令及公开信息层级适配
    final_2001.py              最终论文标量事实、MeV/m适配和3+1 appearance映射
    public_rate_approximation.py 公开DAR总率近似likelihood
  scan.py                      两种模型的profile调度；3+1精确前置缓存；一份活动扫描文件
  core/
    three_plus_one.py          3+1参数与短基线概率
    one_plus_three_plus_one.py  1+3+1参数与短基线概率
    profile_three_plus_one.py  原3+1约束最小化与扫描坐标
    profile_one_plus_three_plus_one.py  原1+3+1约束最小化
    likelihood.py             参考协方差工具、当前协方差和二次型
    calibration.py            Gaussian矩近似与经验Toy CLs；旧二次型反演仅供研究复核
  output.py                   批次/来源路径、调用指纹、共享谱渲染、参数图、轮廓比较、CSV/JSON写入
  paths.py                    唯一仓库定位
```

保留两种模型的不同profile过程，不把它们改造成同一个优化算法。
合并文件不等于修改数值函数；原参数、浮点运算、缓存、求解、优化和随机序列保持。

## 调用链

```text
run.py scan
  -> scan.py
  -> experiments/microboone/adapter.py 读取 configs/analyses，创建选中实验
  -> experiments/microboone 输入、kernel、预测
  -> core/likelihood.py 当前预测协方差和目标函数
  -> core/profile_*.py 固定坐标下的最小化
  -> core/calibration.py
       analytic: 两个固定假设 -> 解析T均值/方差 -> Gaussian尾概率
       toy: 逐点profile观测数据 -> 固定预测和协方差 -> 生成并评价伪数据 -> 经验尾概率
  -> output.py 保存及绘图
```

scan.py 保留 model-specific 调度分支，避免改变优化顺序、默认种子及边界。
输出选择通过已有结果列完成，不在输出中重新计算统计量。
3+1全网格的观测数据profile和广义二次型结果按活动源码、配置、科学输入内容、网格与profile模式生成SHA-256缓存键，保存为可检查CSV和JSON清单。Toy数量、种子、批大小和自适应选择范围不进入此键；它们变化时复用观测数据前置结果并严格重建该点固定假设。缓存读取使用round-trip浮点解析并校验CSV哈希。任何相关输入内容变化都会产生新键，不覆盖旧缓存。
独立profile点使用`scan-workers`线程并行，但`executor.map`保持原笛卡尔网格顺序；每一点仍调用同一个33点盆地搜索、同一边界与同一抛光容差。
逐点高斯/Toy研究位于studies/three_plus_one_toy_distribution_fit/compare_covariance_gaussian.py：
只读旧Toy、调用原asymptotic_cls并重放固定假设对照，绘图调用output.py的plot_statistic_calibration，不进入活动统计推断。
run.py通过begin_output_batch设置文件归组；output.py统一生成结果目录，并在调用结束后记录provenance。
不改变物理参数；实际保存路径与历史结果迁移见OUTPUTS.md。比较入口显式接收四个结果目录。

```text
run.py miniboone --kind official|scan
  -> experiments/miniboone/official_2020.py 读取合作组纯文本发布
  -> official: 不重算，直接保存/绘制官方190x190似然面及覆盖率轮廓
  -> scan: 调用3+1短基线P(mu->e)，按逐事件P*w/N构造两极性信号，合并60维协方差为38维，计算chi2+log|M|
  -> output.py 统一路径、CSV/JSON和参数空间图
```

MiniBooNE尚未注册进MicroBooNE联合adapter。这样先验证该实验自己的公开统计模型，
不会把实验间独立性或相关性作为未经确认的默认假设。

```text
run.py lsnd --kind official|core-mapping|rate-scan
  -> experiments/lsnd/final_2001.py 读取论文中逐项转录的标量记录
  -> experiments/lsnd/public_rate_approximation.py 以DAR谱、IBD相空间和公开几何平均任意appearance概率
  -> official: 写出来源事实表；没有官方数值面或轮廓可重绘
  -> core-mapping: 调用3+1短基线P(mu->e)，以MeV/m转换核对有效appearance振幅
```

LSND final paper的四变量逐事件似然所需事件、PDF和背景变化输入没有公开为数值包，
故该入口不属于联合fit注册项，也不实现一个伪造的Gaussian/Poisson替代。

```text
run.py spectrum
  -> bnb.py / joint.py 组装原预测和参考曲线
  -> output.py 的同一个 render_microboone_spectrum_panels
  -> 图片、逐bin CSV、metadata
```

## 配置在哪里

| 设置 | 位置 |
|---|---|
| 纳入哪些实验、是否允许近似、相关组 | configs/analyses/*.yaml |
| 基线、锚点、输入位置 | configs/experiments/microboone/*/analysis.yaml |
| Fig3a/Fig3b具名运行范围 | configs/runs/fig3a.yaml、fig3b.yaml |
| 模型、analytic/toy/adaptive选择 | run.py 命令行 |
| 3+1网格、Toy数/种子/进程/批次、前置缓存开关/位置 | scan.py 命令行选项；用 --engine-help |
| 1+3+1质量网格、优化迭代/种群/容差 | 同上，--model 1+3+1 --engine-help |
| 当前NuMI曝光比例和准备常量 | numi.py 的 FLUX_*、EVENTS_* 常量 |
| 谱的两个参考点 | bnb.py 的 BNB_PLOT_*；joint.py 的 JOINT_PLOT_*、FIGURE1_* |

准备常量仍保留原值和位置语义，没有借迁移重新选择物理设置。
改变基线/参考点等必须同时审查配置、准备常量、已有kernel metadata；改一个文件不能自动证明旧kernel仍匹配。
preset 先读，显式命令行选项后读；后者覆盖相同选项，须查看最终 metadata。

## 实验接口

每个实验只在自己的 `experiments/<name>/adapter.py` 处理特异输入和公开数据层级；
公共边界声明在 `experiments/interface.py`。扫描、校准、写入和绘图不得复制进实验适配器。
MicroBooNE adapter保留原类型和函数语义，归并选择/组合/注册文件，不新增多层注册系统。
当前注册：
- microboone.bnb.four_channel
- microboone.bnb_numi.joint_four_channel

独立验证入口：`miniboone.nue2020.combined`（尚非联合fit注册项）。
受限审计入口：`lsnd.final_2001`。`rate-scan`可计算模型可移植的总率近似likelihood，但不是LSND四维事件likelihood，也尚非联合fit注册项。

公共需要的是 predict_counts(parameters)、observed_counts、covariance_for_prediction(prediction)、chi2(parameters)。
独立实验可相加；同一 correlation_group 必须以含交叉协方差的联合贡献接入。
新实验需在自己的目录实现输入和预测，再在 adapter.py 接入两个模型需要的分支。
不是只写 YAML 就自动支持任何实验。

## 数据生命周期

- shared/raw：不可改写的14通道公开总谱、背景、观测和系统协方差。
- bnb/inputs：已有可见BNB flux；raw_response：旧公开Reco。
- bnb/derived：归一化及26x60 Reco、参考总协方差审计输入。
- numi/inputs/flux_components：PDF恢复的8份FHC/RHC分flavor输入。
- numi/derived/paper_figure3_weighted_flux：4份曝光平均文件；kernel构建明确读未振荡列。
- numi/derived/public_dk2nu_energy_baseline：公开RHC dk2nu给出的条件基线形状与现有通量边缘结合的登记输入；active BNB+NuMI联合分析读取它并对NuMI振荡概率作能量、味道相关的基线平均，固定基线只作诊断对照。
- 各束流 reweighting：真能量、固定背景、八个过程矩阵、metadata和闭合表。
- 扫描直接读kernel；不重新提取PDF或重建kernel。
- 当前协方差不是固定参考总矩阵：保留分数系统误差缩放，再加当前Pearson对角项。
- 联合分析取完整208x208块，不丢BNB–NuMI交叉块。
- 官方大型NuMI扫描TXT只供 studies/official_grid_profile，不进入本地CLs扫描。
- `NumiFourChannelPredictor`只有在调用者显式传入能量—基线CSV时才以逐源味道的条件质量平均振荡概率；现有事件kernel已经包含flux，因此该路径丢弃psi的能量边缘，禁止重复乘flux。
- MiniBooNE raw发布含11+8+11+8观测bin、两份逐事件全转换样本、60x60分数
  协方差、官方似然面和校准轮廓；未从图片数字化，也未使用MicroBooNE kernel。
- MiniBooNE扫描的有效振幅用sin²theta14=1/2、sin²theta24=sin²(2theta_mue)
  选择appearance简并族中的一个代表，因此核心中的精确4|Ue4|²|Umu4|²等于扫描坐标。
  该选择不影响P(mu->e)，但不得用于声称已重建公开包没有定义的disappearance通道。
- LSND只保留论文明确打印的标量与来源清单；其5697事件、四变量PDF、background
  variation、数值曲面/轮廓都不是本地输入。LSND的3+1映射同样选取
  `sin2(theta14)=1/2`、`sin2(theta24)=sin2(2theta_mue)`，仅保证appearance
  振幅恒等式，绝不导出disappearance预测。

## Profile与校准

3+1 Fig3a固定质量差和精确4*s14*(1-s14)*s24，profile允许的s14并推导s24。
Fig3b固定质量差和4*s14*(1-s14)，检查两支s14并分别profile s24。
s14-profile保留原固定质量差、s14后优化s24的实现。

1+3+1固定两个质量坐标，profile四个模平方和相位；下态有符号质量差为负，上态为正。
保留原幺正可嵌入约束和退耦/零混合边界，不额外引入模型参数。

两种校准均使用既有T和右尾CLs定义。无Toy使用Gaussian分布近似，不是取消profile；
有Toy时每个扫描点只profile观测数据一次，并固定得到的3nu/4nu预测与协方差。每点缓存两套Cholesky分解；每批伪数据保持原随机流，通过多右端三角求解一次性计算全部Toy的两个二次型之差。Toy内部禁止重新profile。有限Toy仍有抽样误差，不代表统计精确无误差。
独立全局prefit及其测试原件保存在 frozen/before_four_block_layout_20260903。
活动表不再带依赖全局prefit的delta_chi2和全局最优点诊断。

## 归档

studies 是一次性研究；frozen 是旧实现，二者不混用。
旧脚本和旧说明完整保留在本次冻结快照；不维持旧导入路径。
ksquare工具已移到 studies/chi_square_gui。
BNB历史数组提取工具移到 studies/bnb_flux_provenance，日常只读已有flux。
PDF提取研究保留原件、算法及来源，不成为扫描导入依赖。

旧结果不删除。读取旧CSV时按metadata区分统计方法、是否全Toy、坐标和模型，不按目录名推断。

## 跨系统运行

活动代码中的仓库路径统一由 `src/sterile_fit/paths.py` 根据模块位置解析，入口
`run.py` 根据自身位置加入 `src`；Windows盘符、登录后的当前目录和服务器用户名
都不参与科学输入定位。数据metadata中保留的历史绝对路径只是来源记录，不作为
活动读取路径。

`scripts/server/run_relics2.sh` 是 Linux 固定入口：它从脚本位置寻找仓库，默认使用
`~/data/venvs/sterile-py311`，设置非交互绘图后端，并把每个进程的BLAS线程默认限制
为1。`scripts/server/sync_relics2.ps1` 是 Windows 同步入口，只接受干净的已提交工作树，
依次更新 GitHub、服务器私有 bare 仓库和服务器 sparse-checkout 工作副本。服务器仅
检出活动 `src`、`configs`、`data` 与根入口；归档、研究工具、测试和输出不进入工作目录。
