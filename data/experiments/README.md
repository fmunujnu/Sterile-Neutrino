# 可复用科学输入

目录内容是输入，不是可随意删除的一次性输出。本次结构整理未修改数值数据或路径。

每个探测器分 shared 和各 beam；各 beam 内分 inputs、raw_response、derived、reweighting。
shared/raw 保留公共发布；derived 和 reweighting 是可复用输入。运行生成物写 outputs。
不同束流若暂用相同响应，必须显式标记借用近似，不能描述为各自准确的探测器MC。

当前路径、准备命令和使用范围统一见根目录 README.md 与 docs/ARCHITECTURE.md。
数据变更必须更新来源/哈希、配置、输入说明和相关回归；详见 docs/MAINTENANCE.md。

LSND 的 `lsnd/` 只保存最终论文逐项标量、来源清单和缺失输入的明确记录；没有图像数字化数据或
派生似然输入。见 `lsnd/README.md`。

MiniBooNE 使用 `miniboone/shared/raw/official_nue2020_combined` 保存合作组直接发布的
纯文本输入、似然面和覆盖率轮廓；来源与逐文件SHA-256见该目录的清单。
