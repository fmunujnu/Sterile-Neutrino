# 当前架构与调用说明

此文描述会随代码变化的事实，不是长期规则。最后结构更新：2026-09-03。
长期规则见根目录 AGENT.md；同步责任见 MAINTENANCE.md。

## 四块

```text
run.py
src/sterile_fit/
  experiments/microboone/
    public_data.py             通道定义、完整公开谱/协方差读取、两个束流选择
    response.py                原始Reco读取、零列/归一化、26-bin适配与准备
    bnb.py                     BNB kernel、预测、构建、闭合、谱数据组装
    numi.py                    NuMI kernel、预测、曝光加权、构建
    joint.py                   联合208-bin协方差、原缓存预测、联合/双面板谱组装
  adapter.py                   配置选择、实验构建、模型接入、Toy观测/假设适配
  scan.py                      两种模型的profile调度；一份活动扫描文件
  core/
    three_plus_one.py          3+1参数与短基线概率
    one_plus_three_plus_one.py  1+3+1参数与短基线概率
    profile_three_plus_one.py  原3+1约束最小化与扫描坐标
    profile_one_plus_three_plus_one.py  原1+3+1约束最小化
    likelihood.py             参考协方差工具、当前协方差和二次型
    calibration.py            解析矩近似与经验Toy CLs，两种函数
  output.py                   批次/来源路径、调用指纹、共享谱渲染、参数图、轮廓比较、CSV/JSON写入
  paths.py                    唯一仓库定位
```

保留两种模型的不同profile过程，不把它们改造成同一个优化算法。
合并文件不等于修改数值函数；原参数、浮点运算、缓存、求解、优化和随机序列保持。

## 调用链

```text
run.py scan
  -> scan.py
  -> adapter.py 读取 configs/analyses，创建选中实验
  -> experiments/microboone 输入、kernel、预测
  -> core/likelihood.py 当前预测协方差和目标函数
  -> core/profile_*.py 固定坐标下的最小化
  -> core/calibration.py
       analytic: 两个固定假设的精确一二阶矩 -> 高斯尾概率近似
       toy: 逐点profile观测数据 -> 固定预测和协方差 -> 生成并评价伪数据 -> 经验尾概率
  -> output.py 保存及绘图
```

scan.py 保留 model-specific 调度分支，避免改变优化顺序、默认种子及边界。
输出选择通过已有结果列完成，不在输出中重新计算统计量。
逐点高斯/Toy研究位于studies/three_plus_one_toy_distribution_fit/compare_covariance_gaussian.py：
只读旧Toy、调用原asymptotic_cls并重放固定假设对照，绘图调用output.py的plot_statistic_calibration，不进入活动统计推断。
run.py通过begin_output_batch设置文件归组；output.py统一生成结果目录，并在调用结束后记录provenance。
不改变物理参数；实际保存路径与历史结果迁移见OUTPUTS.md。比较入口显式接收四个结果目录。

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
| 3+1网格、Toy数/种子/进程/批次 | scan.py 原命令行选项；用 --engine-help |
| 1+3+1质量网格、优化迭代/种群/容差 | 同上，--model 1+3+1 --engine-help |
| 当前NuMI曝光比例和准备常量 | numi.py 的 FLUX_*、EVENTS_* 常量 |
| 谱的两个参考点 | bnb.py 的 BNB_PLOT_*；joint.py 的 JOINT_PLOT_*、FIGURE1_* |

准备常量仍保留原值和位置语义，没有借迁移重新选择物理设置。
改变基线/参考点等必须同时审查配置、准备常量、已有kernel metadata；改一个文件不能自动证明旧kernel仍匹配。
preset 先读，显式命令行选项后读；后者覆盖相同选项，须查看最终 metadata。

## 实验接口

adapter.py 内保留原类型和函数语义，归并选择/组合/注册文件，不新增多层注册系统。
当前注册：
- microboone.bnb.four_channel
- microboone.bnb_numi.joint_four_channel

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
- 各束流 reweighting：真能量、固定背景、八个过程矩阵、metadata和闭合表。
- 扫描直接读kernel；不重新提取PDF或重建kernel。
- 当前协方差不是固定参考总矩阵：保留分数系统误差缩放，再加当前Pearson对角项。
- 联合分析取完整208x208块，不丢BNB–NuMI交叉块。
- 官方大型NuMI扫描TXT只供 studies/official_grid_profile，不进入本地CLs扫描。

## Profile与校准

3+1 Fig3a固定质量差和精确4*s14*(1-s14)*s24，profile允许的s14并推导s24。
Fig3b固定质量差和4*s14*(1-s14)，检查两支s14并分别profile s24。
s14-profile保留原固定质量差、s14后优化s24的实现。

1+3+1固定两个质量坐标，profile四个模平方和相位；下态有符号质量差为负，上态为正。
保留原幺正可嵌入约束和退耦/零混合边界，不额外引入模型参数。

两种校准均使用既有T和右尾CLs定义。无Toy是分布近似，不是取消profile；
有Toy时先逐点profile观测数据，再固定该点的3nu/4nu预测与协方差获得经验分布；Toy内部不重复profile。有限Toy仍有抽样误差，不代表统计精确无误差。
独立全局prefit及其测试原件保存在 frozen/before_four_block_layout_20260903。
活动表不再带依赖全局prefit的delta_chi2和全局最优点诊断。

## 归档

studies 是一次性研究；frozen 是旧实现，二者不混用。
旧脚本和旧说明完整保留在本次冻结快照；不维持旧导入路径。
ksquare工具已移到 studies/chi_square_gui。
BNB历史数组提取工具移到 studies/bnb_flux_provenance，日常只读已有flux。
PDF提取研究保留原件、算法及来源，不成为扫描导入依赖。

旧结果不删除。读取旧CSV时按metadata区分统计方法、是否全Toy、坐标和模型，不按目录名推断。
