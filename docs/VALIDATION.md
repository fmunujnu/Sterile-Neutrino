# 四块结构迁移验证记录

日期：2026-09-03。本文是可变的当前验证记录，不是长期物理原则。

## Unified experiment-adapter boundary (2026-09-06)

- Moved the MicroBooNE, MiniBooNE, and LSND orchestration boundaries to
  `src/sterile_fit/experiments/<experiment>/adapter.py`; removed their former
  package-top-level modules without compatibility shims.
- Added `experiments/interface.py` as the shared likelihood and optional binned
  prediction contract. No probability, covariance, profile, calibration,
  threshold, random sequence, or plotting arithmetic was changed.
- Updated active imports, tests, study consumers, `run.py`, and architecture
  documentation. Frozen migration evidence was intentionally left unchanged.
- Validation: full suite `143 passed`; `python run.py check --analysis all`
  passed; MiniBooNE and LSND command help loaded through their new adapters.
- This validates imports and unchanged tested behavior, not detector-level or
  publication-level physics agreement.

## LSND final-2001 public-record adapter (2026-09-05)

- Added only `data/experiments/lsnd/`, `configs/experiments/lsnd/`,
  `src/sterile_fit/experiments/lsnd/` and its narrow
  `run.py lsnd` dispatch, and `tests/test_lsnd.py`. MicroBooNE, MiniBooNE and
  the common 3+1 physics core were not modified.
- Source basis: Aguilar *et al.*, PRD 64, 112007 / hep-ex/0104049. The
  manifest records exact paper pages: Eq. (1.1) p.2 gives eV2,m,MeV units;
  Sec. IX.B--E pp.23--24 gives the 5697-event four-variable likelihood and
  Gaussian-weighted background variation; Sec. IX.F pp.24--25 says full FC
  generated-data construction was not followed and final regions use
  constant slices; Fig.27 p.64 states Lmax-L<2.3/4.6.
- The implementation verifies only the deterministic identity
  `sin2(2theta_mue)=4|Ue4|^2|Umu4|^2`, MeV/m conversion, zero appearance
  amplitude, printed scalar values, and parsable official/core-mapping output.
  It does not calculate an LSND likelihood, surface correlation/RMSE, contour
  displacement, covariance property, or coverage/Toy result.
- The final publication does not provide a machine-readable event table,
  four-dimensional signal/background PDFs, nuisance/background inputs,
  covariance, likelihood grid, or contour coordinates. No publication image
  was digitised. A local PDF download was not retained after its request was
  cancelled, so the manifest leaves its hash blank rather than inventing one.
- Executed: `python -B -m pytest -q -p no:cacheprovider --basetemp
  tmp/lsnd_pytest_20260905b tests/test_lsnd.py` (6 passed), followed by the
  final source-manifest check `tmp/lsnd_final_pytest_20260905` (7 passed);
  `python -B run.py lsnd --kind official --batch lsnd_final_20260905`;
  `python -B run.py lsnd --kind core-mapping --batch lsnd_final_20260905`;
  the full test command with an isolated `tmp/lsnd_full_pytest_20260905`
  basetemp (138 passed); and `python -B run.py check` (all declared existing
  checks passed). All three LSND test temporary directories were removed after use.

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
历史命令（仅作记录，冻结代码不再执行）：`frozen/studies/structure_migration/check_parity.py`。
```

Windows环境下系统临时目录曾有权限问题，因此本轮使用独立workspace basetemp。
完成后清理自己创建的临时目录；不得用宽泛删除命令清理tmp或仓库根目录。

## 未执行和不能推出的结论

- 未运行完整61×61扫描或大样本Toy；小网格图用于接线验证，不用于排除结论。
- 未运行完整1+3+1质量平面；该模型覆盖源码比较、原单元测试、代表点预测/目标函数和零混合闭合。
- 未重新下载数据、提取PDF、训练kernel或更改统计近似。
- 保留原经验锚点、固定聚合背景、NuMI借用响应及真能支持截断的科学限制。

## 2026-09-12 NuMI公开dk2nu能量—基线输入

- 新NuMI适配路径在零混合时与原固定基线逐bin闭合；非零混合测试确认两者不同。该路径现用于active BNB+NuMI联合分析，固定基线仅保留为诊断对照。
- 缓存的3+1解析基线平均与逐L-bin调用共享核心概率比较，`rtol=atol=2e-14`通过。
- 定向测试：`8 passed`。完整NuMI-only官方100x61格点比较已运行，使用原profile和固定`Delta chi2=5.99`，未使用CLs或Toy。
- 该结果仍借用BNB响应、冻结聚合背景，并以旧公开RHC dk2nu及FHC电荷共轭代理构造条件基线分布，不能解释为合作组正式输入。

## 2026-09-12 非Toy活动校准改为Gaussian

- `analytic`活动入口不再调用广义二次型特征函数反演，改为固定3nu/4nu假设下T的解析均值、方差及Gaussian右尾；输出列改为`p_value_*_gaussian`和`cls_gaussian`，缓存schema升至2以禁止误读旧结果。
- 扫描层只构造一次全网格共享的固定3nu假设；核心Gaussian公式保持冻结实现。一个208-bin联合代表点中，Gaussian校准约0.0087秒，旧二次型反演约0.271秒；两者CLs不同是方法改变，不是数值误差。
- 旧二次型函数只保留给独立历史验证，不被非Toy主扫描调用。
- 非Toy profile现会兑现`--scan-parallel-backend processes`。64点小网格因进程启动从约7.79秒增至约9.76秒；256点联合网格由串行22.61秒降至2进程14.53秒（约1.56倍），两种运行的全部数值列逐点完全一致。故多进程只建议用于中大型网格。
- 完整测试`145 passed`，统一`run.py check`通过。
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
# 2026-09-04 3+1 profile与二次型前置缓存

- 修改范围：仅3+1网格调度、派生缓存和记录；没有改变概率、预测、协方差、profile目标/边界/容差、二次型反演或CLs定义。
- 定向测试：`python -B -m pytest -q -p no:cacheprovider --basetemp tmp/pytest_cache_optimization tests/test_profile_likelihood.py tests/test_scan_plotting.py`，缓存与调度相关测试通过。
- 实际入口：BNB-only 8x8解析小网格首次运行写入缓存，第二次命中缓存；两份结果CSV用round-trip解析后DataFrame完全相等，最大数值差为0。
- Toy控制复用：相同8x8网格改为adaptive-toy、Toy数7、选择带0.005--0.2且诊断点上限0，命中同一缓存，前Toy阶段0.8秒。
- 全套测试：127项中126项通过；唯一失败是本轮前已存在的冻结布局测试仍要求旧的“每个Toy重新profile”函数体，而活动代码已经采用“观测数据profile后固定”策略，本轮未修改calibration.py。
- `python -B run.py check`通过全部BNB、联合协方差、概率、基线、kernel闭合及1+3+1零点检查。
- 尚未运行：61x61生产扫描；小网格性能不能直接外推为完整网格加速倍数。

# 2026-09-04 固定扫描点与批量Toy二次型

- 澄清后的活动策略：3+1与1+3+1均只profile每个扫描点的观测数据一次；Toy内部禁止重新profile。
- 每个扫描点缓存固定3nu/4nu均值、协方差Cholesky分解；同一批Toy用多右端三角求解批量计算T，概率模型、随机流、尾计数和有限样本修正不变。
- 前置内容寻址缓存继续复用观测profile及二次型选择面；改变Toy数量或范围无需重跑全网格前置阶段。
- 208维、1000个人工向量的纯T计算对照：批量路径约4.49倍于逐事件路径；最大绝对浮点差3.55e-15，未改变尾部判定。该微基准不包含伪数据生成、I/O或扫描调度，不能直接等同完整运行加速倍数。
- 全套测试128项通过；包含固定统计量批量/逐事件一致性、相同种子在不同batch size下的一致性和缓存校验。
- 按用户要求未运行真实大规模Toy或生产扫描。

# 2026-09-05 MiniBooNE 2020平行验证入口

- 新增MiniBooNE ν+反ν combined公开发布的只读纯文本副本、直接URL与逐文件SHA-256；未数字化图片、插值或重分bin。
- 新适配器独立读取11+8+11+8观测bin、两份逐事件全转换样本和60x60分数协方差；按发布说明折叠为38维并加入signal统计对角项。MicroBooNE活动物理代码未因本功能修改。
- 定向测试3 passed：输入形状、官方36100点似然面最小值、L/E单位、信号振幅线性、38维协方差对称正定。
- `python -B run.py miniboone --kind official`实际运行完成，输出官方发布曲面及官方频率学派覆盖率轮廓。
- `python -B run.py miniboone --kind scan`实际运行完整190x190网格；使用逐事件P*w/N与`chi2+log|M|`。网格对齐后相对似然面与官方曲面的Pearson相关系数约0.99999975，RMSE约0.233，最大绝对差约1.046；官方与本地网格极小值相邻但不完全相同。
- 上述高度一致是公开Gaussian NLL重建验证，不证明本地已复现合作组的频率学派覆盖率生成；本地尚未实现MiniBooNE fake-data coverage，也未注册跨实验联合fit。
- MiniBooNE逐事件概率随后改为调用活动3+1短基线appearance核心；有效振幅的确定性分解严格满足4|Ue4|²|Umu4|²等于官方横坐标。改动前后36100点完整曲面的最大绝对变化3.21e-11、平均绝对变化1.10e-12，最优网格点不变，确认只有浮点舍入级差异。
- 3+1接入后定向测试7 passed；全套132 passed且统一`run.py check`通过。没有把公开包无法识别的P(mumu)/P(ee) disappearance加入MiniBooNE控制样本或背景。
- 从已重算的36100点曲面增加无热力图轮廓叠加：本地使用二维固定似然阈值2.30/4.605/9.210/11.83，官方点保持发布的coverage校准坐标；该图是方法差异诊断，不把两类线称为同一统计量。

# 2026-09-13 MiniBooNE完整网格10000-Toy校准

- relics1批次`mb_full_10000_half`完成全部36100个参数点，28个worker无缺片。
- 每个被检验点生成10000份Toy；每份Toy均在相同190x190网格上重新profile。
- 合并表含36100个唯一`tested_point_index`，每行Toy数均为10000，保存于
  `data/experiments/miniboone/shared/derived/reprofile_toy_10000/point_calibration.csv`。
- `python run.py miniboone`现默认读取该表，生成本地Gaussian NLL、官方likelihood、
  官方频率学派轮廓与本地reprofile-Toy轮廓的有/无热力图对比。
- 该结果仍是基于公开38维Gaussian模型的本地校准，不冒充合作组内部校准。

# 2026-09-05 MiniBooNE 90% coverage 5000-Toy试验

- 在官方90%轮廓选择6个代表点，每点5000份、共30000份38维Gaussian Toy；使用各点协方差的Cholesky因子抽样。
- 每份Toy在完整190x190（36100点）发布网格上重新寻找最优NLL，并把通常不落在网格节点的被检验真点本身加入候选；全部Toy统计量非负。
- 官方轮廓点所需的观测相对NLL约4.988--5.320；固定二维90%阈值为4.605；本地Toy 90%临界值约4.901--5.894。
- 按与官方所需临界值的绝对差计，Toy在6点中的4点更接近，固定阈值在2点更接近。5000-Toy bootstrap区间显示部分差异超过纯Toy抽样波动，不能声称Toy实现已复现官方coverage。
- 主要未决项是合作组未完整公开的fake-data细节及连续全局拟合与本地发布网格拟合的差别；该研究不进入活动推断。首次使用NumPy通用多元正态采样的试运行因高条件数协方差触发警告，已判无效并删除，未用于上述结果。

# 2026-09-06 MicroBooNE沿排除线分块的低Toy位移试验

- 沿既有Fig. 3a/3b固定Toy排除线各切分6个连续质量区块；每块采样3个质量位置和5个振幅位置，每个生成假设使用50份Toy。
- 共评估180个扫描点和18000份逐Toy profile，实际耗时997秒；活动扫描、预测和协方差实现未修改。
- 局部平面仅在估计的CLs=0.05根落入实际采样窄带时用于平移；超出采样带的区块标记为`unresolved_outside_sample_band`，禁止外推。
- Fig. 3a有2/6区块得到带内粗估计，Fig. 3b有3/6区块得到带内粗估计；其余区块在50 Toy精度下不足以定位交点。
- 该结果只用于估计方向和量级，不是coverage校准的正式排除线。

# 2026-09-06 MicroBooNE排除线尾部稀疏性探针

- 不使用区块或边界拟合，直接沿既有Fig. 3a和Fig. 3b排除线各取24点；每点、每个生成假设使用50份Toy。
- Fig. 3a的4nu尾计数中15/24为零，均值0.542；Fig. 3b中16/24为零，均值0.375，两者中位数均为零。
- 在两图共有的约0.18--14 eV2质量范围内，Fig. 3a的4nu尾计数10/12为零、均值0.333；Fig. 3b为15/23为零、均值0.391。该精度不能稳定判断两图4nu尾部谁更高。
- 同一共有范围内，3nu尾计数均非零；Fig. 3a均值10.17、中位数9，Fig. 3b均值14.22、中位数11。这里只能作为相对稀疏性诊断，不是精确概率比较。

# 2026-09-06 逐Toy-profile低统计点态带

- 复用沿既有排除线窄带的180个参数点；这些点的每一份Toy均单独执行profile，每点、每个生成假设各有50份Toy。
- 对已观测尾计数作Jeffreys平滑的二项重复运行重抽样，估计10、20、30、50 Toy下的点态95%交点范围；未重新生成更高统计的逐Toy-profile样本。
- 严格限制到5000-Toy参考值处于CLs=0.005--0.1的采样点后，50 Toy时Fig. 3a仅9个质量点、Fig. 3b仅10个质量点能在已有振幅采样带内产生交点；更低Toy时大量质量点完全无法定位边界。
- 已解析点的上下振幅中位数比在50 Toy时约为Fig. 3a的1.27和Fig. 3b的1.22；该数值受“只有能解析的点才进入汇总”的强选择效应影响，实测宽度没有随Toy数单调缩小，不能据此拟合可靠的1/sqrt(N)缩放。
- 图中5000-Toy参考线使用活动的观测profile后固定假设策略，不是高统计逐Toy-profile真值；低Toy蓝带不能解释为保证覆盖最终逐Toy-profile排除线的同时置信带。

# 2026-09-06 CLs窄带200份逐Toy-profile扫描

- 从5000-Toy参考网格选择CLs=0.01--0.07区域，沿质量方向每隔一个切片保留一个，并在保留切片中计算全部带内振幅点；Fig. 3a为82点，Fig. 3b为151点。
- 每个参数点分别在3nu和4nu生成假设下运行200份Toy，每份Toy单独profile；共保存93200条逐Toy检验量和尾部标记到可检查CSV。
- 每个质量切片在log振幅上对CLs作分段线性插值；对5000次参数化bootstrap得到的交点分布取2.5%、50%、97.5%分位，形成未经平滑的下界、预测线和上界。
- 两图各28个质量切片均得到三条插值线。有效bootstrap交点比例均值为Fig. 3a的0.581和Fig. 3b的0.673；局部折点反映低尾计数、非单调CLs和分支切换，不能解释为物理精细结构。
- 追加逐横排拟合诊断：每个固定质量切片把CLs作为log振幅的一维因变量作线性拟合，并允许交点外推到Toy采样带外、但裁剪到完整扫描坐标范围；不进行二维图像插值或跨质量平滑。
- 横排拟合的中位预测交点在Fig. 3a有7/28个、Fig. 3b有13/28个落在实际Toy振幅范围外；中心斜率方向异常的切片分别为6/28和3/28。bootstrap斜率接近零时上下界会延伸至扫描边界，因此宽蓝线是低统计不可辨识性，不是精确物理区间。

# 2026-09-07 Fig. 3a右下角1000份逐Toy-profile收敛检查

- 在图面坐标0.7<=sin2(2theta_mue)<=1、0.01<=Delta m2<=0.05 eV2内计算完整3x11正式网格，共33点；每点在3nu和4nu下各1000份Toy，每份单独profile。
- 33点和66000条逐Toy检验量全部保存；使用同一随机流的前500、750、1000份作嵌套收敛比较。
- 对每个固定振幅沿log质量分段线性寻找CLs=0.05，并锁定最靠近中心交点的同一分支，避免bootstrap切换到约0.045 eV2处的第二交点。
- 1000-Toy中位交点在三个振幅处分别为0.02351、0.02150、0.02050 eV2；相对500-Toy中位线移动分别为+0.00038、+0.00005、+0.00038 eV2。
- 1000-Toy点态95%全宽分别为0.00221、0.00158、0.00114 eV2；为对应500-Toy宽度的0.79、0.86、0.50。中心线达到约2%量级稳定，但仅三个振幅点，不能证明全局排除线收敛。
- 与正式`quadratic_non_toy_20260904/scan_fig3a_analytic`逐参数profile加二次型校准线比较：正式交点为0.02373、0.02158、0.02034 eV2；1000份逐Toy-profile中位交点为0.02351、0.02150、0.02050 eV2，差值绝对值为0.00022、0.00009、0.00016 eV2。该局部两种统计处理在约1%量级一致。

# 2026-09-06 LSND公开DAR总率3+1 likelihood初步扫描

- 沿用活动3+1短基线appearance概率，不修改核心；新增LSND实验适配层，将任意appearance概率对公开可重建的DAR权重积分。
- 输入为论文发布的平均振荡概率`0.264% +/- 0.067%(stat) +/- 0.045%(syst)`、muon-DAR反muon中微子Michel谱、IBD主导相空间，以及30 m中心距离和8.3 m轴向长度。
- 明确缺少源尺寸、横向几何、能量依赖效率、重建迁移和四维事件PDF；结果是可移植到未来1+3+1概率的rate-only近似，不是Kopp获得的LSND合作组likelihood。
- 241x241扫描共58081点。论文四维best-fit `(Delta m2, sin2(2theta_mue))=(1.2 eV2, 0.003)` 在总率近似中预测平均概率0.002218，对应相对最优`Delta[-2 ln L]=0.274`，与公开总率相容。
- 总率项产生主低质量斜带和高质量平均振荡带；因只有一个观测量，最小值沿曲线退化，不能自行确定官方best-fit、高质量小岛或DAR+DIF形状信息。
- 使用与论文constant-slice相同的二维阈值4.605和9.210绘制90%/99%线；这些是固定likelihood切片，不是该近似的coverage校准结果。
- 验证：`tests/test_lsnd.py`为10 passed；完整测试142 passed；`python -B run.py check`全部通过。输出位于`outputs/lsnd_final_2001/three_plus_one/public_rate_approximation_20260906_v2/`。

# 2026-09-06 LSND最终论文公开分箱谱重加权研究

- 只新增`studies/lsnd_public_spectrum_reweighting/`研究路径；未修改MicroBooNE、MiniBooNE或活动振荡核心，也未把读图结果静默提升为官方输入。
- 直接读取本地LSND最终论文PDF第54页Fig.16和第62页Fig.24的矢量路径，不使用截图像素点选；分别恢复5个正电子能量bin和11个`L/E` bin的beam excess、非束流宇宙线扣除后仍保留的两类中微子背景、低质量差参考信号和非对称误差棒。
- Fig.16矢量提取闭合：beam excess总和50.701，对论文印刷总数49.1的差为1.601；背景总和16.833，对16.9的差为-0.067；参考信号总和32.585，对32.2的差为0.385。程序将这些容差写成fail-fast检查。
- 用低质量差小相位下`P`正比于`(L/E)^2`反推每bin等效kernel形状，并以论文的`33300 * 0.39`（100%转化事件数乘`Rgamma>10`关联光子效率）定标；每bin概率采用48点Gauss-Legendre积分，不再只取bin中心。
- 独立bin Gaussian近似使用图示beam-excess误差；没有公开bin间协方差、背景nuisance或四维事件PDF，因此本结果是可用于未来1+3+1概率的公开资料降维近似，不是LSND官方likelihood或coverage复现。
- 后续按最终论文已公布误差加入两个逐扫描点解析profile的归一化nuisance：信号相对宽度`sqrt(0.10^2+0.07^2)=12.2%`，总中微子背景相对宽度`2.3/16.9=13.6%`。仍未虚构逐bin形状误差或协方差；上一条中的“没有背景nuisance”应读作没有公开的逐bin/分量背景nuisance。
- 241x241的3+1扫描中，论文最佳点附近相对最低点的`Delta chi2`为Fig.16的0.996和Fig.24的1.224，均通过预设`<2.3`内部相容检查。两套90%轮廓在低质量差主带接近；高质量差起伏对分箱敏感，不判作可靠物理细节。
- 实际运行完成；`tests/test_lsnd.py`在工作区独立临时目录下10 passed，研究目录`git diff --check`通过。默认pytest缓存目录仍因既有Windows权限锁产生非物理警告。
- 输出：`outputs/studies/lsnd_public_spectrum_reweighting/latest/`，包含两份提取CSV、两份重绘谱、两份扫描CSV、各自参数空间图、轮廓对比图和机器可读验证metadata。

# 2026-09-13 NuMI psi活动登记与Fig.3b范围

- 确认活动BNB+NuMI适配器已经固定读取`public_dk2nu_energy_baseline/psi_exposure_weighted_four_flavours.csv`；本次只统一配置、注释和文档状态，没有改变既有预测数值。
- NuMI组件明确登记为`active_joint_approximation`，不注册成独立NuMI实验。psi的能量边缘不重复乘入已经吸收flux的经验事件kernel。
- Fig.3b具名preset的质量平方差上界由14改为40 eV2，因此新运行会实际计算到40，而不是只扩展空坐标轴。
- 定向测试11 passed，`python -B run.py check`全部通过。一次较宽的测试集合另暴露既有冻结AST一致性失败及Windows pytest临时目录权限问题；二者与本次psi配置变更无关，未冒充全套通过。
- 10 m基线bin中心采样与bin内均匀解析平均的诊断差异：在测试混合`s14=s24=0.05`下，14 eV2时逐bin最大约0.81%，40 eV2时约1.73%。这表明扩展高质量区需要基线离散化收敛检查；该均匀bin比较未替换活动算法。
- 实际运行Fig.3b无Toy Gaussian近似：质量轴按原0.1--14 eV2的对数步长密度由61点扩展为74点，振幅轴保持61点，共4514个逐点profile；全部`chi2`和`cls_gaussian`为有限值。结果位于`outputs/microboone_bnb_numi_joint/three_plus_one/fig3b_psi_formal_gaussian_40ev2_74x61/scan_fig3b_analytic/`。14--40 eV2的快速条纹尚未通过psi基线bin细化收敛检验。
- 新增独立研究运行器`studies/microboone_fig3b_full_reprofile_toy/run.py`及服务器分片器：74x61每点在3nu和4nu下各生成指定数量的Toy，每份Toy重复固定图坐标下的完整`s24`/两支`s14` profile。1点、每假设2 Toy烟雾测试通过并已清理临时输出；100 Toy配置仅用于定性交叉验证。

# 2026-09-13 每Toy profile首次计算缓存

- 性能剖析确认重复成本来自不同Toy在同一扫描点访问相同profile网格参数时，反复生成相同预测、缩放相同协方差并执行相同Cholesky分解。
- 新增每个扫描点局部、容量256的LRU缓存，只保存由振荡参数决定的预测和Cholesky因子；每份Toy仍使用自己的观测残差，profile网格、两支物理解、精修算法、容差、随机数和尾判据均未改变。
- 固定种子的1点、每假设2 Toy前后CSV逐项完全相同，Toy统计量最大绝对差为0。新增单元测试验证不同Toy观测共享参数工作时仍与未缓存二次型严格相同。
- 本机连续3点、每点每假设10 Toy的稳定计算部分约0.3秒/点；1点、每假设100 Toy的完整进程运行约5.0秒。测试只在本机执行，未同步或运行服务器版本。

# 2026-09-13 relics1 Fig.3b逐Toy profile全网格

- 提交`7c676c9`在relics1完成74x61共4514点，每点在3nu和4nu下各100 Toy；28个worker全部完成、0失败，墙钟时间49分18秒。
- 下载后验证4514个全局点索引连续且唯一，各点Toy数声明一致；合并结果位于`outputs/microboone_bnb_numi_joint/three_plus_one/fig3b_reprofile_toy100_relics1/result.csv`。
- 在`0.03 <= CLs <= 0.07`的547个近边界点中，4nu右尾计数中位数为0、3nu右尾计数中位数为24。100 Toy时有限样本比值在典型点只能从约0.04跳到0.08，故轮廓中的锯齿和孤立闭环主要是Toy量化噪声，不能解释为已解析的物理结构。
- 新增分片合并绘图器，输出0--1线性色标热图、纯轮廓图和机器可读计数诊断；原始服务器分片保持不变。

# 2026-09-13 Fig.3a+Fig.3b 5000-Toy任务准备

- 逐Toy profile研究运行器扩展为可选`fig3a`、`fig3b`或`both`：Fig.3a保持61x61、`1e-4--1`和`0.01--100 eV2`；Fig.3b保持74x61、`0.01--1`和`0.1--40 eV2`，联合共8235个扫描点。
- 生产默认改为每个生成假设5000 Toy，即每个参数点总计10000份Toy；每份Toy的重新profile、随机种子派生和统计定义不变。
- 原先逐Toy逐次打开CSV改为每个参数点一次批量追加，数值和抽样次序不变，避免5000 Toy任务被文件打开开销主导。
- 本机分别对联合索引0的Fig.3a点和索引3721的Fig.3b点完成每假设2 Toy烟雾测试；两图模式、局部/全局索引和输出字段正确。正式5000-Toy全网格未运行。

# 2026-09-14 MicroBooNE 5000-Toy正式结果接入

- relics2正式批次`mb_fig3ab_reprofile_5000`完成：28个worker全部成功，Fig.3a为61x61、Fig.3b为74x61，共8235个唯一网格点；每点在3nu和4nu生成假设下各5000 Toy，且每份Toy重新执行固定图坐标下的profile。
- 仅下载合并后的逐点校准摘要，没有把体积巨大的逐Toy诊断表提升为活动输入。规范表位于`data/experiments/microboone/shared/derived/reprofile_toy_5000/`。
- 新增活动入口`python run.py scan --model 3+1 --calibration reprofile-toy`。它严格校验点数、网格、Toy数和有限值后只负责统一绘图，不重复高开销计算；Fig.3a和Fig.3b均成功生成热力图、纯排除线和CSV。
- 活动输出位于`outputs/microboone_bnb_numi_joint/three_plus_one/primary_reprofile_toy_5000/`。该结果仍继承公开输入的NuMI探测器响应近似，不等同于合作组内部完整模拟。

# 2026-09-19 NuMI基线分布默认规则

- 修正研究对比中的隐式旧固定基线输入：NuMI默认预测、谱图、3+1与1+3+1联合适配均使用登记的`q(L|E,nu)`条件基线分布。
- 固定0.680 km路径仍保留，但只能通过显式`--include-fixed-baseline-comparison`诊断选项使用，输出必须带`fixed_baseline`名称。
- BNB加NuMI卡方对比器只接受重新生成的`fig3a/fig3b_energy_baseline.csv`，不再读取历史`fig3*_local_numi_only.csv`。
- 已实际完成100x61的Fig.3a与Fig.3b能量—基线加权NuMI-only profile，固定基线选项未启用；输出位于`outputs/studies/numi_only_official_comparison/energy_baseline_current/`。随后仅复用保存面重绘BNB+官方NuMI与BNB+本地NuMI的固定`Delta chi2=5.99`诊断图。
- 相对旧固定基线面，Fig.3b沿质量轴的profile分支切换由561次降至382次，`sin2_theta24`大于0.25的相邻跳变由47次降至5次；说明大量锯齿确由固定基线近似放大，但剩余高质量结构仍受离散E/L中心采样、借用BNB响应、3--5 GeV缺失和固定聚合背景限制。
- 定向验证13项通过，`python -B run.py check`全部通过，`git diff --check`通过。
- 已用活动联合208-bin协方差、NuMI `q(L|E,nu)`加权路径完成无Toy Gaussian CLs扫描：Fig.3a为61x61，Fig.3b为74x61且质量上限40 eV2；结果位于`outputs/microboone_bnb_numi_joint/three_plus_one/energy_baseline_gaussian_cls/`，另从相同CSV生成纯轮廓图，未重新计算统计量。

# 2026-09-21 MiniBooNE appearance-only 1+3+1无Toy固定质量切片

- 未改动现有MiniBooNE 3+1概率、38-bin折叠、参数相关协方差或Gaussian NLL；新增实验私有的1+3+1逐事件appearance适配器，逐条使用公开full-transmutation MC的`E_true`、`L_true`、`E_QE`和`weight/N`。
- 以`A4=4|Ue4|^2|Umu4|^2`、`A5=4|Ue5|^2|Umu5|^2`为扫描坐标；对称最小行范数分解仅用于现有核心的幺正可嵌入检查。中微子/反中微子采用相反CP相位号，每个扫描点连续profile一个appearance CP相位。
- 事件概率与现有1+3+1核心逐事件标量调用在12个公开事件上最大差异为0；预计算信号基底与直接重加权完整中微子/反中微子事件表的逐bin最大差异约`2e-11`事件。
- 正式运行三个固定质量切片`(0.1,1)`、`(1,1)`、`(1,10) eV^2`；每个切片为61×61的`A4,A5 in [3e-4,1]`对数网格，共11163点。CSV坐标无重复，全部数值有限，三个切片均跨过4.605和9.210固定阈值。
- 输出位于`outputs/miniboone_nu_nubar_combined/one_plus_three_plus_one/miniboone_1p3p1_non_toy_20260921/gaussian_nll_fixed_mass_slices/`，含总表、逐切片best fit、best-fit预测、三联及单切片热力图和纯轮廓图。
- 这些线是各固定质量切片相对自身最小值的二维渐近阈值，不是Toy、MiniBooNE官方coverage或完整七维1+3+1全局排除。公开包未分解的电子背景及缪子控制样本没有被虚构成disappearance成分。
- 定向`tests/test_miniboone.py`为7 passed；统一`python -B run.py check`全部通过。全套测试在仓库内独立临时目录为151 passed、2 failed；两项失败都是本轮开始前已修改的MicroBooNE NuMI/联合类与2026-09-03冻结AST不一致，本轮未改这两个文件。系统默认pytest临时目录仍有既有Windows权限锁，改用显式basetemp后不再产生19项环境错误。

# 2026-09-21 MiniBooNE appearance-only 1+3+1质量profile卡方面

- 正式默认入口已从固定质量切片改为外层扫描`A4,A5`、内层profile`|dm2_41|,dm2_51,phi_mue`；旧切片只保留为显式诊断模式。
- 外层为31×31的`A4,A5 in [3e-4,1]`对数网格；两个质量差分别在`[1e-2,1e2] eV2`的21点对数网格上profile。相位先用5点粗筛全部质量对，再对最优10个质量对连续细化。
- 第一阶段严格只用`chi2=(D-M)^T V(M)^-1(D-M)`；参数相关协方差沿用现有MiniBooNE重建，刻意不加`log(det(V))`，不运行Toy。
- 正式表为961行、坐标无重复、全部数值有限；`delta_chi_square`范围为0到约27.44，并跨过4.605和9.210。网格最小值为`chi2=23.180828`，位于`A4=0.066943`、`A5=0.197435`、`|dm2_41|=0.251189 eV2`、`dm2_51=0.158489 eV2`、`phi=0.049964 rad`；两个质量差都不在profile范围上界。
- 输出位于`outputs/miniboone_nu_nubar_combined/one_plus_three_plus_one/miniboone_1p3p1_profiled_chi2_20260921/chi_square_profiled_mixing_plane/`。绘图复用MiniBooNE 3+1的共享对数坐标、色表和轮廓样式，而不是另写一套风格。
- 当前轮廓中的小岛和折线包含21点离散质量profile及31点外层网格的分辨率效应；在加密/连续质量优化之前，不能把这些细小结构解释成稳定物理特征。

# 2026-09-21 MiniBooNE图示固定质量、CP守恒卡方面

- 按后续澄清停止把质量profile面作为默认结果；该面及旧固定切片只保留为诊断。活动默认固定`dm2_41=-0.9 eV2`、`dm2_51=0.5 eV2`、`phi_mue=0`。
- 扫描坐标改为参照图中的`|Ue4 Umu4|`和`|Ue5 Umu5|`，各自在`[1e-3,0.15]`取61点对数网格；概率内核使用严格映射`A_i=4|Uei Umui|^2`。
- 在appearance-only公开likelihood中，固定乘积后单独的电子/缪子矩阵元分解完全不可识别；该方向的profile解析平坦，不引入会改变卡方的任意代表值或额外nuisance。
- 仍只计算参数相关协方差的二次卡方项，不加入`log(det(V))`、不运行Toy。正式表共3721行、坐标无重复、全部有限，`delta_chi_square`范围为0到约448.21并跨过4.605和9.210。
- 网格最小值为`chi2=40.320620`，位于`|Ue4 Umu4|=0.005313`、`|Ue5 Umu5|=0.042862`；对应`A4=0.000113`、`A5=0.007348`。
- 输出位于`outputs/miniboone_nu_nubar_combined/one_plus_three_plus_one/miniboone_1p3p1_fixed_masses_cp0_20260921/chi_square_product_plane/`。热力图和纯轮廓图调用MiniBooNE共享对数画布、色表、阈值线和最优点样式。
- 该结果只含MiniBooNE appearance公开项，不能与参照图中的全球`app.`、`disapp.`或`all`区域直接等同。
- 定向MiniBooNE测试为11 passed，`run.py check`全部通过，`git diff --check`通过。全套为155 passed、2 failed；两项仍是本轮开始前已经存在的MicroBooNE NuMI/联合类与旧冻结AST不一致，本轮没有修改对应实现。

# 2026-09-22 活动profile接口与研究脚本整理

- 新增纯声明对象`ProfileSpecification`，要求每个3+1或1+3+1真实模型参数明确归入固定值或profile边界；不改变概率、目标函数、优化算法、容差或随机种子。
- 1+3+1固定质量对活动profile已用显式声明表示完整体积、4态脱耦和5态脱耦候选；零混合边界仍单独精确评价。3+1的派生有效振幅扫描保持原专用物理约束实现。
- 核对联合适配器：3+1与1+3+1都通过`NumiEnergyBaselineDistribution`调用登记的`q(L|E,nu)`；固定NuMI基线没有进入活动联合预测。
- `chi_square_gui`和已完成的结构迁移工具移动到`frozen/studies/`。所有MicroBooNE/BNB/NuMI复杂分析、Toy及调试研究目录保持原位未动。
- profile定向测试13项通过，`python -B run.py check`通过，`git diff --check`通过。完整测试使用仓库内独立临时目录后为157 passed、2 failed；两项仍是本轮开始前的冻结AST守卫没有登记NuMI条件基线类/联合类改动，不是本轮profile接口测试失败。
