# Toy分布研究（非活动推断）

使用当前 adapter.py、core/profile_three_plus_one.py、core/calibration.py 和扫描种子规则。
在随机但固定种子的Fig3a点，记录同一Toy的chi2_4nu、chi2_3nu及其差T。

比较非中心卡方、独立非中心卡方之差等边际近似；两项真实卡方由同一Toy计算，通常相关，
不能把独立差分布解释成真实组成项。使用同一样本拟合后给出的普通KS p值未做拟合参数校准，
只作探索性诊断，不能当作严格拟合优度或尾概率验证。

```powershell
python studies/three_plus_one_toy_distribution_fit/run.py --points 4 --toys 200 --workers 2
```

默认不超过两进程，结果在 outputs/studies/three_plus_one_toy_distribution_fit/<UTC批次>/results。
不进入正式扫描；不把此研究拟合出的参数自动写回核心。
主接口变更时更新脚本导入及此说明；统计方法改变另需说明证据和局限。

## 不重新profile：比较协方差高斯与已有Toy

```powershell
python -B studies/three_plus_one_toy_distribution_fit/compare_covariance_gaussian.py --samples outputs/studies/three_plus_one_toy_distribution_fit_low_resource/legacy/results/toy_samples.csv --metadata outputs/studies/three_plus_one_toy_distribution_fit_low_resource/legacy/results/metadata.json --analysis-config configs/analyses/microboone_bnb_numi.yaml --batch gaussian_check_new
```

显式选择样本/metadata/分析配置；拒绝覆盖同批次。适配原Fig3a研究的样本列，不支持任意未知格式。
输出为outputs/studies/three_plus_one_toy_distribution_fit/<batch>/gaussian_comparison。
共享output.py默认恢复每点一张图：两列对应3nu/4nu，三行是密度、右尾、CDF偏差。
第三排为F_Toy-F_Gaussian（不是p值直方图）；灰带为单个预先指定CDF的95% DKW带，半宽sqrt(log(40)/(2N))，不作8组同时覆盖声明。
绿色为profiled Toy，灰虚线为同一伪数据的固定假设对照；接近重合时可判断这些点的profile影响。
--layout combined仍可生成此前一张合图；不删除旧图和样本。gaussian_deviations.csv记录均值偏差、宽度比、KS最大CDF偏差与右尾差。
红线：核心asymptotic_cls的均值/方差，不拟合Toy。绿线：已有每Toy重新profile结果。
固定假设对照：原种子、原协方差重放同一组高斯伪数据，只计算固定假设T；保存在表中。
重放时逐份核对原chi2_3nu、原观测点chi2_4nu以及T列关系，不重新执行profile或换种子。
原始样本仍只读；附加固定T及高斯p写samples_with_gaussian_p.csv；所有参数和KS/尾区间写distribution_comparison.csv。
observed_cls_comparison.csv给出当前高斯CLs与原样本的加一尾估计比值。

这里p不是被假设为高斯的变量。高斯近似针对T；p_H(t)=P(T>=t|H)。
若连续参考分布准确，在同一H下生成样本并转换出的p应服从[0,1]均匀分布，而不是正态分布。
KS检验使用预先由协方差决定的参数，没有用本次Toy拟合，因此不同于旧脚本的naive fitted-KS。
各组KS p未作多重比较校正，不等于模型排除p或CLs；未拒绝不证明高斯准确。
200样本、零尾事件的95%双侧Clopper–Pearson区间为[0,0.0183]，不能分辨千分之一量级尾概率。

## 扩大为10000份与快速重绘

```powershell
python -B studies/three_plus_one_toy_distribution_fit/run.py --points 4 --toys 10000 --workers 2 --samples-only --output-directory outputs/studies/three_plus_one_toy_distribution_fit/toy10000_20260903/samples
```

两种假设各10000份，合计80000份。参数点种子和Toy种子沿用原默认值。
--samples-only仅跳过不相关的卡方分布族拟合，不跳过任何Toy的profile；每500份打印进度。
完成的参数点/假设即时保存checkpoints，最后toy_samples.csv和metadata.json完整保存。
检查点仅是故障时可恢复的数值记录，目前不提供自动续跑；不要把不完整检查点当成最终样本。
随后用比较脚本显式传入该samples目录内的CSV/metadata和原分析配置，--batch使用同一批次名。
生成的原始样本、附加p值样本、统计摘要均保留，不需为改图重新模拟。

```powershell
python -B studies/three_plus_one_toy_distribution_fit/compare_covariance_gaussian.py --plot-only outputs/studies/three_plus_one_toy_distribution_fit/toy10000_20260903/gaussian_comparison
```

--plot-only只读完成的CSV，默认生成/覆盖四张point_XX_gaussian_vs_toy.png及偏差表；附加--layout combined才覆盖合图。
不生成伪数据、不profile、不重新计算协方差。

### 均值和方差为什么这样取

固定3nu/4nu预测为m_0,m_1，构造chi2用V_0,V_1；生成假设H的均值、协方差为m_H,V_H。
令D=m_H+Lz、LL^T=V_H、z服从N(0,I)，则固定假设差可以严格写成

$$T=c+b^Tz+z^TBz,$$

$$B=L^T(V_1^{-1}-V_0^{-1})L,$$

$$b=2L^T[V_1^{-1}(m_H-m_1)-V_0^{-1}(m_H-m_0)],$$

$$c=(m_H-m_1)^TV_1^{-1}(m_H-m_1)-(m_H-m_0)^TV_0^{-1}(m_H-m_0).$$

因此核心代码使用的两个矩是精确的（前提：高斯伪数据、固定预测和协方差）：

$$\mu_H=c+\mathrm{tr}B,\qquad\sigma_H^2=2\mathrm{tr}(B^2)+b^Tb.$$

实际实现用Cholesky线性求解，不显式求逆。把整个分布取为N(mu_H,sigma_H^2)才是近似步骤。
均值是位置参数，不叫非中心卡方的“非中心参数”；高斯只有均值和方差两个参数。
若V_0=V_1，B=0，T为高斯变量的线性组合，固定假设下严格正态。
若V不同，二次项存在，一般不严格正态；多个不占支配地位的模态累加或线性项主导时可近似，不能只因208bins就保证。
每Toy重新profile后，最优预测和协方差依赖样本，T不再是同一个固定二次型，上述矩不再保证精确。
本次四点灰绿几乎重合，是这些点的数值事实，不可推广到所有点或未来模型。

KS参考：[SciPy官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html)。

### 高阶矩的可行路线（讨论，未改变核心）

仍在上述高斯伪数据、固定二次型前提下，三四阶累积量可直接算，不需要从Toy拟合：

$$\kappa_3=8\operatorname{tr}(B^3)+6b^TBb,$$
$$\kappa_4=48\operatorname{tr}(B^4)+48b^TB^2b.$$

偏度为kappa3/sigma^3，超额峰度为kappa4/sigma^4；第四中心矩为kappa4+3sigma^4，不能混淆。
它们来自累积量生成函数K(t)=ct-(1/2)log det(I-2tB)+(t^2/2)b^T(I-2tB)^(-1)b在零点求导。
所以已知均值、协方差、固定假设的T结构，就已知高阶累积量；若不假设伪数据高斯，仅知道协方差则不足以确定高阶矩。
Edgeworth/Gram–Charlier可作便宜修正，但截断后可能出现负密度或不单调CDF，尾部不保证改善。
更值得先研究鞍点近似（使用完整K）或特征函数数值反演/广义卡方算法，而非只往高斯补有限阶项。
参考[Kuonen 1999, pp.929–935](https://doi.org/10.1093/biomet/86.4.929)、[Imhof 1961, pp.419–426](https://doi.org/10.1093/biomet/48.3-4.419)。
它们解决固定二次型分布，不自动解决每Toy重新profile的分布。profile还需局部二次近似/显式处理参数边界与分支，
再用现有代表点Toy交叉验证，不能把固定假设的精确计算冒充profile精确校准。
