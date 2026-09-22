# 输出布局与历史结果

## 日常结果

路径为 `outputs/<source>/<model>/<batch>/<product>/`。

| 层级 | 当前名称及含义 |
|---|---|
| source | microboone_bnb、microboone_numi、microboone_bnb_numi_joint、miniboone_nu_nubar_combined；公共输入图为microboone_public |
| model | three_plus_one、one_plus_three_plus_one；不涉及模型的输入检查为inputs |
| batch | 自动UTC微秒时间戳，或用户指定的 --batch 标签 |
| product | spectra_bnb、spectra_joint、spectra_figure1；scan_fig3a_analytic、scan_fig3b_toy、scan_mass_pair_analytic、chi_square_fixed_masses_cp_zero等；质量profile诊断为chi_square_profiled_mixing_plane，旧固定切片诊断为gaussian_nll_fixed_mass_slices；另有public_plots、flux_plots、prepared_events、contour_comparison |

联合结果只存联合来源，不复制到BNB和NuMI两处。其他实验或多实验联合使用自己的来源标签。
扫描仍原样保存表、图、metadata以及可选toy_distributions，不进一步拆散相互配套的文件。
不同模型的CSV文件名和列名保持原有约定；不是通过改名转换模型。

```powershell
python run.py spectrum --kind figure1 --batch comparison_01
python run.py scan --preset fig3a --calibration analytic --batch comparison_01
python run.py scan --preset fig3b --calibration analytic --batch comparison_01
```

上述扫描为完整网格示例，不是快速检查。Toy设置照原命令，只增加批次标签即可。
批次标签是文件归类，不是Toy抽样批量，不参与随机数种子或统计计算。
标签只允许字母、数字、下划线、点、连字符，禁止路径和Windows保留名。
同一批次下的provenance逐次追加，不覆盖先前记录；记录传入计算入口的参数及活动源码、配置文件哈希。
它不是代码快照，也不保证未登记的数据/环境可完整复现；实际方法及完成状态看产物metadata。
失败后已经产生目录的运行也可能有provenance，不能把它视为完成标记。

## 显式选择旧结果重绘

compare 不再隐式寻找process24或run1/run2。当前比较器仍比较原来的解析和adaptive混合CLs列，未改变绘图和统计方法。

```powershell
python run.py compare --batch replot_01 --fig3a-analytic outputs/microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3a_analytic --fig3b-analytic outputs/microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3b_analytic --fig3a-toy outputs/microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3a_adaptive-toy --fig3b-toy outputs/microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3b_adaptive-toy
```

`--source` 只指定比较图的来源标签，不选择实验、不改变计算。
comparison_sources.json保存输入目录。显式输出路径仍优先，适合临时检查。

## 研究工具

独立研究输出使用 `outputs/studies/<研究名>/<batch>/results/`，不伪装成正式扫描。
工具代码仍在studies，输出仍集中outputs。共享 `result_directory("studies", 研究名, "results")` 生成路径；
需要归入指定批次时先调用 `begin_output_batch(标签)`；通过 `finish_output_batch(实际参数列表)` 可记录批次provenance。
历史工具暂不自动写统一provenance，保留自己的metadata。未改变它们的科学算法。
高斯/Toy比较工具使用同一研究目录下的gaussian_comparison产物，支持--batch并记录provenance；
其中distribution_comparison.csv为统计摘要，samples_with_gaussian_p.csv为可复查逐样本表；
当前默认每点一张point_XX_gaussian_vs_toy.png，包含密度、右尾、CDF偏差；gaussian_deviations.csv为偏差表。
支持--plot-only仅从CSV重绘；附加--layout combined保留all_points_gaussian_vs_toy.png合图选择。

## 已有输出迁移

- process24的解析、adaptive Fig3a/3b和轮廓比较集中在联合来源的legacy_process24批次。
- 历史谱归入各自来源的legacy_spectra批次；NuMI原目录混有输入图和事件表，保留为mixed，避免虚构更精确来源。
- 原checks移入studies对应研究名的legacy/results。
- 原archive/legacy_runs整体归入legacy/unclassified/imported/results，未猜测内部来源。
- outputs/migration_20260903.json记录每次目录移动、每个文件的SHA256和校验状态，支持反向查找。
- 历史metadata中的旧绝对路径保持原文；解读时配合迁移表，不重新写成“当年就在新位置”。
- 旧空目录壳清理；既有archive/locked_test_cache权限锁定，未绕过权限处理。

批次legacy_process24是原运行名的归组，不代表本次使用24进程。既有结果没有重新扫描。
以后输出布局、命令或消费者接口变化，必须同步本文件、README和输出测试。
# 独立二次型校准输出（2026-09-03）

`outputs/studies/quadratic_toy_check/<batch>/results` 保存选择坐标、原始Toy、
二次型系数/曲线、尾计数和CLs比较、6个单点图。均为CSV/JSON/PNG。
沿用`plot_statistic_calibration`；新增可选`panel.candidate={T,pdf,sf}`，
仅此研究提供该字段。旧调用无字段时布局与计算不变；新图第三行改为相对候选CDF残差，
正态仅作淡色背景，不用于研究p-value/CLs。详见该study README。

后续`assessment`产物复用已存样本：第三排同轴加入Gaussian-vs-quadratic和Toy-vs-Gaussian残差，
另存KS/尾概率/T阈值/振幅边界位移CSV。Gaussian作为对照独立计算，不反馈到候选CF或Toy推断。
`plot_calibration_boundaries`是共享输出层的研究用切片布局，不替换主扫描图。
其可选amplitude_label指定坐标标签；默认保留mue，ee研究显式传入ee，不改变数值或主扫描绘图。
二次型研究的mode及Toy混合参数保存规则、历史样本复核目录reuse_verification，详见studies/quadratic_toy_check/README.md。
