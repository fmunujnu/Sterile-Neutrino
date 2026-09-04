# NuMI 当前数据说明

这是近似分析输入，不是官方未发布MC。本目录不依赖 outputs 中的文件。

- inputs/flux_components：8份FHC/RHC × nue、numu及反中微子CSV；0--5 GeV，50个0.1 GeV bin，单位/POT/cm²/100 MeV。
- 来源是MicroBooNE Note 1129蓝色New Flux曲线的本地PDF矢量提取；不是MINERvA束流替代，也不是官方ROOT数组。
- derived/paper_figure3_weighted_flux：4份曝光平均文件，保留未振荡列及两个参考参数列。目录名是历史名称，不代表新的Fig3参数定义。
- 曝光权重仍为FHC 0.308、RHC 0.692，准备代码单基线仍为0.680 km。
- 参考谱参数仍为Δm²41=1.2 eV²、sin²(2θμe)=0.003、sin²θ24=0.018或0.0045。
- kernel构建只读取未振荡曝光加权列；不会重复施加参考振荡。
- 读取发布通道8--11，每通道26 bins；单束流协方差是4×4个26×26块，即104×104。
- 借用已有BNB 26×60 Reco；将每个100 MeV输入bin按原方法拆给两个50 MeV真能bin。
- Reco真能支持只到3 GeV，PDF flux延伸到5 GeV；该截断与冻结聚合背景的近似保持不变。
- reweighting 保存8个过程kernel、真能量、固定背景、闭合和metadata。

当前联合分析已经通过 joint.py 使用完整BNB–NuMI交叉协方差，不再说“联合分析尚未接入”。
但单独NuMI扫描ID未注册，联合模式仍标记approximate，不能称为合作组完整复现。

日常检查：python run.py inputs --kind numi-flux。
显式重建：python run.py prepare --kind numi-flux，然后 --kind numi-kernel。
结构迁移没有执行重建。变更输入时按 docs/MAINTENANCE.md 更新来源、哈希、准备说明和回归。
