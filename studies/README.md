# 一次性研究与检查

这里不是日常运行入口。活动profile、谱和比较统一使用根目录 run.py。
旧实现放 frozen；一次性研究留在这里，不能成为活动预测的依赖。

| 目录 | 用途 |
|---|---|
| numi_flux_pdf_extraction | PDF矢量提取、原图重绘、已存数组对照；保留输入来源复现链 |
| numi_energy_baseline_kernel | 生成公开NuMI dk2nu约束的psi(E,L)登记输入；活动分析读取data中的固定快照，研究脚本只保留来源复现链 |
| microboone_fig3b_full_reprofile_toy | 0.1--40 eV2、74x61完整Fig3b网格；每点在3nu/4nu下生成Toy并对每份Toy重新profile的研究校准 |
| official_grid_profile | 官方大网格的profile/Wilks对照，不是本地CLs |
| three_plus_one_toy_distribution_fit | 少量点的Toy分布拟合，默认最多两进程 |
| quadratic_toy_check | 独立固定二次型CF反演，对照相同Toy的固定与profiled T；不替换活动推断 |
| cls_plus_b_comparison | 已结束的CLs+b对照，不是活动统计方法 |
| scan_result_comparison | 指向frozen内的原轮廓脚本；日常改用 run.py compare |
| bnb_flux_provenance | 从冻结基准语法解析flux的来源工具，不执行冻结代码 |

已结束且不再运行的`chi_square_gui`和`structure_migration`保存在
`frozen/studies/`。MicroBooNE/BNB/NuMI相关的复杂profile、Toy、粗糙度和输入诊断
本轮均原位保留，等待单独科学审查后再决定是否归档。

每个研究目录README仅解释自身用途、依赖和输出。接口变化时必须更新直接导入的函数；
当前主架构不在这里复制，统一见 docs/ARCHITECTURE.md。
输出统一为outputs/studies/<研究名>/<批次>/results，详见docs/OUTPUTS.md。
此前结果的legacy批次只作历史证据，新增小工具同样复用output.py路径接口。
