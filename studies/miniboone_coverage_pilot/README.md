# MiniBooNE full-grid reprofile Toy study

这是非正式研究版本，不改变 `run.py miniboone --kind official/scan`。

默认对公开的完整 `190 x 190` 参数网格逐点校准，每个被检验参数点生成
`10000` 份 Toy。观测数据先在完整网格上 profile 一次；随后每份 Toy 都在
同一个完整网格上重新 profile，而不是固定观测数据得到的最优参数。

结果逐点追加到 `point_calibration.csv`。`--start-point` 和 `--stop-point`
用于分段提交；每点随机流由主 seed 与扁平参数点编号共同派生，因此分段方式
不会改变该点的 Toy 样本。

该实现使用公开的38维 Gaussian 模型和有限网格，不等同于 MiniBooNE 合作组
内部的完整频率学派校准。

本次仅完成代码修改、未执行 Toy 计算；运行时间和最终覆盖边界尚未验证。
