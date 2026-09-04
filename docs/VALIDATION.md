# 四块结构迁移验证记录

日期：2026-09-03。本文是可变的当前验证记录，不是长期物理原则。

## 变更边界

- 活动运行代码由原src的40个Python文件和scripts的20个文件，归并为src的19个文件（含包标记和paths）及根入口run.py。
- 分为实验特异输入、适配/调度、核心计算、共享输出；一次性工具保留在studies，旧实现保存在frozen/before_four_block_layout_20260903。
- 独立全局prefit入口及实现已归档。活动profile保持原优化器、参数边界、分支、种子、容差和计算顺序。
- 不再生成仅服务于旧全局prefit的delta_chi2、best_fit_found、prefit_seeds诊断；逐点chi2、T、profile参数和CLs保留。
- 未重建kernel/flux/Reco/协方差，没有修改科学输入或既有输出。
- 统一谱渲染包含原四通道和上下双面板布局；扫描图保留原绘图算法，只移动到共享输出模块。

## 已执行

1. 迁移前主测试：73 passed。
2. 迁移后主测试：92 passed。
   - 原prefit边界测试原件随快照归档，活动测试改为零振幅profile边界检查。
   - 新增13组原数值源码AST比较，以及统一入口、移除prefit、输出和布局测试。
   - AST检查读取冻结源码，不导入或执行冻结代码。
3. python -B run.py check：BNB输入/响应/闭合、当前协方差、概率守恒通过；联合208-bin输入/交叉协方差和1+3+1零混合闭合通过。
4. studies/structure_migration/check_parity.py：67份科学输入哈希及20组代表点数值在迁移前后完全一致。
   - 3+1与1+3+1概率，包括中微子/反中微子。
   - BNB和联合预测、完整协方差、目标函数。
   - Fig3a、Fig3b各一个固定点的profile最优参数和chi2。
   - 解析CLs以及每假设8份、单进程、固定种子的profiled Toy统计结果。
   - 基准和逐项报告现位于outputs/studies/structure_migration/legacy/results/before.json、parity.json（输出布局迁移只改变路径）。
5. 实际入口小网格：Fig3a和Fig3b解析2×2扫描，以及3+1 Toy 2×2扫描通过；生成CSV、metadata、profile图。
6. 同一2×2 Toy任务分别单进程、双进程运行；result.csv与4份逐点Toy分布CSV逐字节一致。
7. 实际谱入口：BNB四通道加两参考参数、上下BNB/NuMI nue CC FC图成功生成；双面板已做目视布局检查。
8. git diff --check通过；没有提交或推送仓库。
9. PDF提取研究自己的9项测试通过；其数值代码未改变。

## 重跑方式

从仓库根目录运行：

```powershell
python -B -m pytest -q -p no:cacheprovider --basetemp tmp/check_unique_name
python -B run.py check
python -B studies/structure_migration/check_parity.py --baseline outputs/studies/structure_migration/legacy/results/before.json --output-directory outputs/studies/structure_migration/new_check/results
```

Windows环境下系统临时目录曾有权限问题，因此本轮使用独立workspace basetemp。
完成后清理自己创建的临时目录；不得用宽泛删除命令清理tmp或仓库根目录。

## 未执行和不能推出的结论

- 未运行完整61×61扫描或大样本Toy；小网格图用于接线验证，不用于排除结论。
- 未运行完整1+3+1质量平面；该模型覆盖源码比较、原单元测试、代表点预测/目标函数和零混合闭合。
- 未重新下载数据、提取PDF、训练kernel或更改统计近似。
- 保留原经验锚点、固定聚合背景、NuMI借用响应及真能支持截断的科学限制。
- 少量Toy不验证尾概率精度；前后相等不等于合作组精确复现或全参数空间正确。
- 既有根目录.pytest_cache和outputs/archive内旧缓存有权限锁，未绕过权限处理。

未来任何代码改动都应按MAINTENANCE.md更新这里，不能把本轮通过状态当作永久保证。

## 随后的输出布局整理（2026-09-03）

- 已迁移既有输出：22组路径、246个文件。目录对应及逐文件SHA256见outputs/migration_20260903.json；全部记录状态verified。
- 仅改路径、批次记录和历史结果选择接口；核心数值函数AST对照仍通过，科学数据未更改。
- 主测试102 passed；独立PDF研究测试9 passed；统一run.py check通过。
- 显式读取迁移后的before.json，67份输入哈希和20组数值（含每假设8个Toy）仍严格相等。
- 实际执行figure1谱、解析Fig3a 2×2扫描；相同layout_check_20260903批次正确归组，产生provenance。
- 比较入口读取迁移后的四组历史CSV成功生成原有11张图及comparison_sources.json，不重新拟合；11张PNG与历史版本逐字节一致。
- 新路径/非法批次/无输出时不写provenance有单元测试。完整扫描及数据重建未运行。
- 默认解析和Toy方法、图的统计定义与原曲线计算不变；没有以迁移结果宣称新的物理准确性。

## 逐点协方差高斯与旧Toy比较（2026-09-03）

- 找回4个Fig3a随机点，每点每假设200份，合计1600份已有profiled Toy；未新增profile拟合。
- 当前代码重建各点原预测与协方差，重放原种子的高斯数据；全部逐Toy的3nu chi2与存档在rtol=1e-10、atol=1e-8内一致，原观测4nu chi2亦一致。
- 红线参数直接调用核心asymptotic_cls，无Toy拟合；灰线为同一伪数据的固定假设T，绿线为原profiled T。
- 8组profiled T对预先指定高斯的KS p为0.0614–0.8188，未作多重比较校正；不构成尾概率精度或全局适用性证明。
- 点2、3的4nu右尾计数均为0/200，加一修正p=1/201；不能用这样的样本量区分高斯预测的约0.001尾概率。
- 四图、逐样本p、KS摘要及CLs表位于outputs/studies/three_plus_one_toy_distribution_fit/gaussian_check_20260903/gaussian_comparison；四图已目视检查，随后改善大数坐标刻度显示。
- 主测试105 passed；统一run.py check通过。未改核心概率、profile、协方差或CLs算法，未运行完整扫描。

## 扩大到10000 Toy并合图（2026-09-03）

- 同一4个扫描坐标，两种生成假设各10000份，总计80000次逐Toy profile；两工作进程，BLAS/OMP线程各1；模拟约1204秒。
- 使用研究入口--samples-only，只跳过原不相关的卡方分布族拟合，不跳过profile；完成组即时保存8份checkpoint，最终原始统计量CSV完整保留。
- 本轮逐样本null chi2的协方差重放核验通过；8组各10000个连续唯一toy_index，CSV总行数80000。
- 与旧200份前缀并非全部严格相等：点0/1误差约1e-12；点2观测profile略变，预测相对最大差6.74e-10、协方差1.14e-9；点3选到互补简并分支，预测与协方差相对差小于6e-16。
- 旧前缀更严格容差检查曾失败，原因和逐项差异保存在legacy_prefix_audit.json；不将其写成全体逐位一致，也不以修改旧样本使检查通过。
- 主测试107 passed，统一check通过；新合图测试和plot-only不构建物理对象的测试通过。
- 合图只含高斯/重新profile Toy的密度与右尾；每列一个点，4行对应两假设的两种显示，无p值直方图。实际plot-only从缓存CSV再绘得到逐字节相同PNG。
- 输出批次outputs/studies/three_plus_one_toy_distribution_fit/toy10000_20260903，samples与gaussian_comparison分开放置；旧200份结果保留。
- 10000份后8组profiled分布的KS检验均可检出对指定高斯的差异，不能沿用小样本时“未检出差异”的结论。点2/3在4nu下仍为零右尾计数，不解释为真实尾概率严格为零。

## 四张图与累计偏差（2026-09-03）

- 仅从10000份批次的已保存CSV重绘；没有新增Toy/profile、没有改校准算法。
- 默认恢复四张独立图，第三行为F_Toy-F_Gaussian残差及95%单CDF DKW波动带；不恢复p值直方图。合图通过--layout combined保留。
- gaussian_deviations.csv新增均值差、标准差比、右尾差，沿用已计算KS与原样本量。8组最大CDF偏差约1.84–4.28个百分点；N=10000的单CDF波动带半宽约1.36个百分点，不作8组联合覆盖声明。
- 主测试108 passed、run.py check通过；新增独立图适配器与偏差列测试。代表性点2图已目视检查。
- 高阶矩、鞍点、广义卡方只是研究说明中的后续建议，未接入核心；固定假设矩不能冒充每Toy重新profile后的精确矩。
# 固定二次型CF独立验证（2026-09-03）

- 新研究`studies/quadratic_toy_check`，未接入默认扫描；核心目录逐文件SHA256与运行开始一致。
- Gaussian仅在既有共享绘图器作为淡红背景；新p/CLs由CF反演计算，不从Toy拟合系数。
- `pilot200_20260903`用旧100 Toy边界选点，2400次profile约110秒；发现旧边界受尾计数和平滑修正限制，不能当准确边界。
- `boundary200_20260903 --refine-near --toys 200`：三个质量差处以CF法定位0.035/0.05/0.07；另三个远点。单进程/BLAS单线程，2400份逐Toy profile约89.7秒（不含选点定位）。
- 六点CF CLs约0.0350/0.0500/0.0700/0.1887/0.2408/0.00341；Toy平滑估计约0.0889/0.0976/0.1011/0.2500/0.2571/0.0294。
- 前三个点4nu尾计数3/3/8（每组200），精度不足，不声称边界偏差已经确认或高斯已经被全面改进。
- 12个profiled分布KS距离0.03875–0.09616；单CDF 95% DKW半宽约0.09603，非多点联合验证。
- 使用每尾Clopper-Pearson区间(alpha=.05/12)再取比，六点同时保守95%区间约[.00395,.415]/[.00425,.470]/[.0207,.285]/[.00861,1]/[.0436,.953]/[.0000239,.170]；所有区间跨.05，因此此样本量不认证任何排除边界。
- CF网格减半差最大7.5e-15；CDF截断界最大3.91e-10；前者仅收敛检查，不冒充严格积分总误差界。
- 正态线性特例、非中心卡方特例、带正负特征值对称性、与原核心矩匹配及可选绘图接口均测试通过。
- 主测试`python -B -m pytest -q -p no:cacheprovider --basetemp tmp/quadratic_final_tests_20260903`：113 passed；`python -B run.py check`通过。默认系统pytest缓存目录权限失败后改用独立工作区临时目录；不是代码失败。
- 单点图0和2已目视检查；全部原始Toy、系数、曲线与诊断CSV保存。未做大Toy验证、全参数扫描或Fig3b验证；固定二次型仍不能冒充重新profile的精确分布。
# 增量Toy、CDF残差与局部边界位移（2026-09-03）

- `triplets1000_20260903`：12点、24组各1000份，共24000行且(point,generator,toy_index)唯一；约1187.6秒，不含选点与报告。
- `reused10000_20260903`：复用历史80000份样本，核对观测chi2、两个解析矩；没有新Toy。
- 最终报告分别在`reused10000_final_20260903/assessment`与`triplets1000_final_20260903/assessment`；其他assessment批次为中间版本，不作为当前推荐结果。
- 第三排新增Gaussian-Quadratic与Toy-Gaussian曲线，保留Toy-Quadratic；大样本8组KS均减小，典型0.035–0.043降到0.008–0.010。点1仍有profile偏差；不能宣称候选分布精确等于profile后分布。
- 新24组有19组KS减小；所有二次型观测尾概率落入24尾Bonferroni/Clopper-Pearson同时95%区间，不等于证明无偏。
- 三质量差0.029286/39.8107/100 eV²，二次型振幅边界相对Toy插值为-4.2%/-9.1%/-4.2%；高斯真实理论求根相对Toy为+50.2%/+13.7%/+20.6%。
- Toy边界bootstrap区间含二次型边界；区间描述性、只计MC重采样误差、不计三点插值误差和模型误差。低质量差高斯边界在原Toy范围外，额外实际理论计算定位，不作外推。
- 记录固定T阈值的所有向下交叉数、最近交叉、bootstrap有效比例；缺根留空，不静默插补。
- 极小CF尾低于截断界+网格诊断时p/CLs留空，另存原积分数值与数值上限估计；不把1e-14冒充已验证尾概率。
- 新增分布评估/无外推/零尾区间/数值不可分辨负例测试；主测试117 passed，统一check通过；核心文件哈希运行前后相同。
- 未执行全参数扫描、Fig3b边界验证、1 eV²附近的密集边界验证，也未修改主校准方法。

## 两个正式扫描profile的候选分布核验（2026-09-03）

- 研究入口增加显式appearance-profile/electron-disappearance-profile；观测profile调用原核心，每Toy调用正式scan._profile_toy_at_scan_point。不改核心、物理输入、profile优化设置、抽样或主推断。
- Fig3a保留triplets1000_20260903原24000份样本；通过正式接口重放12点×2假设×5份=120份，chi3、profiled chi4及固定T在rtol=1e-10/atol=1e-8内一致；不是全24000份重新模拟。
- 数据文件哈希和样本哈希清单在fig3a_dispatcher_verified_20260903/reuse_verification/metadata.json；旧文件不覆盖。
- Fig3b在三个CF候选边界点和三个历史较远点各生成两假设1000份，批次fig3b_profile1000_20260903；12000行完成且索引唯一，约1614.54秒。历史远点5在当前方法下也接近边界，不将旧标签当成新距离证明。
- 主测试124 passed（tmp/profile_modes_final_tests_20260903）；run.py check通过。新测试核对正式调度器一致、ee两个物理解、振幅约束和非法坐标负例。
- 新Fig3b另外60份通过正式接口重放，chi3/profiled chi4差为0；所有新Toy的振幅约束最大误差4.03e-16，T_profile均不大于同Toy固定T。
- 68份科学输入文本、7个核心文件、Fig3a旧samples/scores哈希未变。两个本轮独立pytest临时目录已清理，旧结果均未覆盖。
- Fig3b二次型对固定T的KS为0.0152–0.0344，对正式profile后T为0.1092–0.3640；12组均超过同时95% DKW/Bonferroni阈值0.05556。高斯KS为0.1081–0.3613；二次型只在1/12组更小。
- profile平均使T下降1.142–3.602；离散分支切换并非唯一原因。固定二次型不能作为这两种profile的统一已验证替代。
- 8/12个二次型观测尾概率落在同时区间外，但6个CLs比值仍在较宽同时区间内；不得混淆CDF失配、比值抵消和边界精度。近边界点的区间跨0.05，未认证完整排除线位移。
- 六张图及完整报告在fig3b_profile1000_20260903/assessment/REPORT.md；运行图点0和评估图点2已目视检查。旧Fig3a报告继续保留，不冒充本轮全部重算。
