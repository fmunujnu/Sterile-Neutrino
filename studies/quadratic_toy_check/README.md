# 固定二次型分布与重新 profile Toy 的独立验证

不改变活动物理、协方差、随机数或 profile，不接入默认 CLs 扫描。
`quadratic.py` 从现有假设构造 T=c+b'z+z'Bz，特征值分解后数值反演特征函数。
使用现有 SciPy Simpson 积分，不安装新的分布拟合包；不从 Toy 拟合任何参数。
参考 Imhof (1961), https://doi.org/10.1093/biomet/48.3-4.419 。
常数与一次项保留，允许负特征值；无正态替代、无独立两卡方假设。
固定二次型的精确分布仍只是重新 profile 后 T 的候选近似。

## 运行

```powershell
python -B studies/quadratic_toy_check/run.py --scan-csv outputs/microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3a_adaptive-toy/result.csv --toys 200 --batch boundary200_20260903 --refine-near
```

历史结果只用于选坐标：3点靠近旧有限Toy边界、3点远离边界；不作为预测输入。
推荐`--refine-near`：在前三个质量差处用新CF法定位CLs=0.035/0.05/0.07，
而非将旧低统计Toy的0.05当作准确边界。这里只决定新验证坐标，不调分布迎合Toy。
保留历史幅度和新幅度；边界搜索用brentq只改变本研究选点，不改变任何profile优化器。
所有6点重新计算观测profile和理论预测；单进程，数值库线程各1；每种假设200份，共2400份。
沿用既有seed派生、Gaussian抽样和逐Toy profile；同一份伪数据额外计算固定T对照。
默认测试Fig3a的appearance-profile；`--mode electron-disappearance-profile`改为正式Fig3b坐标，
必须同时提供含fixed_sin2_2theta_ee列的Fig3b扫描CSV，不能将mue列重命名当作ee。
观测数据与每份Toy均使用正式扫描的核心profile；Toy直接调用scan._profile_toy_at_scan_point。
Fig3a固定质量平方差和A_mue=4*s14*(1-s14)*s24，约束下profile剩余混合。
Fig3b固定质量平方差和A_ee=4*s14*(1-s14)，保留s14=(1±sqrt(1-A_ee))/2两个分支，
各自在s24∈[0,1]上profile后取较小chi2；不把观测数据选中的分支锁死给Toy。
s14/s24在本说明指sin²(theta14)/sin²(theta24)，不是角度本身。

输出：outputs/studies/quadratic_toy_check/<batch>/results。
selected_points.csv记录选择依据；逐组samples即时保存，最终合并samples.csv；
coefficients CSV和scores里的constant可重建每个固定分布；curve CSV可重绘；
scores.csv记录固定/重新profile的KS距离、尾计数、95%二项区间、积分诊断；
cls_comparison.csv只计算二次型和Toy的CLs，正态不参与p-value或CLs计算。
metadata和provenance记录代码/config/source hashes、资源和状态。

图片复用output.py的plot_statistic_calibration可选candidate接口。每点两列为3nu/4nu，
三行为密度、右尾、相对二次型CDF残差；蓝色为二次型，绿色为profiled Toy，灰线为同Toy固定T，
淡红色正态仅背景参照。单CDF DKW带不是多点联合保证，也不是CLs=0.05线。

积分明确记录CDF截断余项界和网格减半差；后者不是严格积分误差上界。
不裁剪Toy，不通过调参让候选分布迎合Toy。初步200份不足以认证排除边界；
零尾计数不代表零概率，有限样本点估计沿用(k+1)/(N+1)。

## 首轮结果

`pilot200_20260903`保留旧低统计边界选点；推荐查看`boundary200_20260903`，
该轮三个near点由新方法重新定位。最终六图`point_00_comparison.png`至`point_05_comparison.png`。
后轮2400次profile约90秒（不含选点定位），113项测试通过，核心文件哈希未变。
near点CF CLs=0.035/0.05/0.07，Toy平滑结果=0.0889/0.0976/0.1011。
4nu尾计数仅3/3/8，现阶段不能证明改善了边界；详见docs/VALIDATION.md的误差范围。

## 增量统计与边界评估

`--boundary-triplets --toys 1000`在3个质量差各定位CF CLs=0.15/0.05/0.015，
另保留3个远点，共12点、每假设1000份。仍单进程且完整profile。

```powershell
python -B studies/quadratic_toy_check/run.py --scan-csv outputs/microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3a_adaptive-toy/result.csv --boundary-triplets --toys 1000 --batch triplets1000_20260903
python -B studies/quadratic_toy_check/assess.py --source outputs/studies/quadratic_toy_check/triplets1000_20260903/results --batch triplets1000_20260903
```

`reuse_large_sample.py --source <旧gaussian_comparison目录> --batch <新批次>`可直接复用
既有10000 Toy和同Toy固定T，重新计算CF曲线；核对当前预测chi2和矩，绝不重新生成Toy。
然后用`assess.py`按相同格式评估；绘图及评估输出在新批次assessment，不覆盖旧结果。

第三排增加橙色`F_Gaussian-F_quadratic`和红色`F_Toy-F_Gaussian`，与原绿色
`F_Toy-F_quadratic`同轴；主扫描绘图没有candidate时不受影响。

评估产物：
- distribution_metrics.csv：两候选KS、固定T的KS、profile影响、KS改进的配对bootstrap区间；不是靠不显著就证明相等。
- tail_cls_comparison.csv：观测处两候选/Toy尾概率及CLs；二项Clopper-Pearson+Bonferroni同时区间。
- T_cut_comparison.csv：同一参数点的CLs(t)=0.05阈值，报告所有向下交叉数并选择最靠近Tobs的交叉；无交叉不外推。
- amplitude_boundary_shifts.csv与boundary_slices.png：每个质量差3个振幅点，插值q=p4-.05p3的零点及对数/百分比位移；保留网格括区，不冒充完整扫描线。

分析bootstrap独立使用种子20260904、300次重采样已存统计量，不调用物理Toy生成。
bootstrap是描述性近似，Tcut/振幅区间注明有效根比例及条件性，不冒充严格95%覆盖。
有多个根时振幅边界不指定唯一值，bootstrap出界比例单独报告；零尾计数仍必须看精确二项上界。
固定阈值的多点尾区间有同时覆盖控制；边界插值误差和候选模型错误不包含在bootstrap内。
高斯比较边界另外用原profile在相邻采样点之间求根，去掉三点插值的高斯侧误差；
核对源配置哈希，不将其他实验配置静默当成BNB+NuMI。Toy侧仍是稀疏插值，须结合括区判断。
如果高斯根不在Toy三点范围内，扩展实际理论计算到振幅1尝试定位，绝不对缺根线性外推；metadata和CSV记录扩展标志。

当前推荐结果：`outputs/studies/quadratic_toy_check/triplets1000_final_20260903/assessment/REPORT.md`；
大样本图：`outputs/studies/quadratic_toy_check/reused10000_final_20260903/assessment`。
原始样本仍分别在triplets1000_20260903/results与历史toy10000_20260903中，未覆盖。
较早的reused10000/checked及triplets1000的assessment是中间诊断版本；最终报告已标记数值分辨率、修正坐标标签，并补齐边界定位。
极小CF尾概率不报告伪精确数值：低于截断界+网格诊断的分辨率则p/CLs留空，保留原积分值和数值上限估计；该数值上限不是严格总积分误差保证。

## 两种正式扫描profile验证

```powershell
python -B studies/quadratic_toy_check/run.py --scan-csv outputs/microboone_bnb_numi_joint/three_plus_one/legacy_process24/scan_fig3b_adaptive-toy/result.csv --mode electron-disappearance-profile --refine-near --toys 1000 --batch fig3b_profile1000_20260903
python -B studies/quadratic_toy_check/assess.py --source outputs/studies/quadratic_toy_check/fig3b_profile1000_20260903/results --batch fig3b_profile1000_20260903
python -B studies/quadratic_toy_check/verify_scan_reuse.py --source outputs/studies/quadratic_toy_check/triplets1000_20260903/results --mode appearance-profile --batch fig3a_dispatcher_verified_20260903
```

新批次metadata记录scan_mode；samples新增toy_sin2_theta14/toy_sin2_theta24，
用于检查Toy是否选择另一分支；scores用generation_sin2_theta14/generation_sin2_theta24标识生成假设。
旧CSV保留原样，复核工具显式识别其旧列名，不修改旧数据。
verify_scan_reuse仅重放各组均匀取样的5份Toy并调用正式profile，检查源代码/config/数据哈希；
这不是全部24000份重新计算，也不产生新的独立统计样本。
Fig3b六点没有每质量差三个振幅，故本轮仅评估逐点分布/尾概率/T阈值，不能由此认证完整排除线位置。
assessment的distribution_metrics.csv还检查每份新Toy的实际振幅约束，记录两侧s14分支计数和s24范围。
相反s14半区计数在Fig3b表示离散分支切换，在Fig3a只表示连续约束曲线跨过s14=.5，不能混称。

必须区分：T_fixed(D)=chi4(D;A,eta_hat_observed)-chi3(D)，
T_profile(D)=min_eta chi4(D;A,eta)-chi3(D)。候选CF只给前者分布。
如果数值profile正确找到了不高于固定点的解，则T_profile<=T_fixed；
两种生成假设下的下降量不必相同，故CLs的比值不保证消除偏差。
不要因Fig3a某些点结果较好就将其作为Fig3b或全扫描的保证。

本轮完成结果：`outputs/studies/quadratic_toy_check/fig3b_profile1000_20260903/assessment/REPORT.md`。
Fig3b新增12000份，固定T与CF的KS为0.015–0.034，正式profile后为0.109–0.364，故候选分布不能直接推广。
保留Fig3a原24000份，并重放120份；Fig3b另重放60份，正式接口结果均一致。
CLs比值有时因误差抵消而接近；本轮未认证整个扫描平面的排除线，也未替换主方法。
