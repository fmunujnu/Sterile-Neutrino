# 修改后的同步清单

此表规定“改了什么，就必须更新什么”。完成代码但未更新对应文档，不算完成。
AGENT.md 管长期原则；其余文件记录可变实现。不要把相同说明复制到多个README。

| 变更 | 必须同步 |
|---|---|
| 入口、命令行、默认参数、运行preset | README.md命令；ARCHITECTURE.md参数表；相关configs/runs；入口测试 |
| 文件移动、合并、删除、导入关系 | ARCHITECTURE.md目录/调用链；活动调用者和测试；归档说明 |
| 实验加入/停用、相关组、近似状态 | configs/analyses；ARCHITECTURE.md注册列表；README科学范围；输出metadata；选择测试 |
| 原始数据版本、路径、单位、通道/bin | 数据来源/哈希；输入说明；适配器；负例与逐bin回归；README科学范围 |
| kernel/Reco/flux准备改变 | 对应实验README或ARCHITECTURE数据生命周期；来源metadata；输入哈希和闭合测试 |
| 概率、参数定义、约束、profile | 原理证据、对应核心注释；ARCHITECTURE；profile/模型回归；VALIDATION.md |
| 协方差、T、CLs、Toy生成或近似 | 核心注释；README方法说明；ARCHITECTURE；结果metadata；统计与随机数回归 |
| 图片布局、列名、保存位置、metadata格式 | output.py公共接口；README结果说明；OUTPUTS.md；输出测试；已有文件迁移哈希/路径表；活动结果消费者 |
| 一次性研究工具 | studies/<purpose>/README.md；只记录该工具用途/依赖/结果位置，不复制主运行教程 |
| 验证完成或发现未解决问题 | VALIDATION.md：命令、范围、结果、未执行内容、限制 |
| 长期协作准则真正改变 | AGENT.md；必要时更新本表，不堆入当前文件清单 |

## 通用完成条件

1. 确认不覆盖无关用户改动。
2. 结构迁移先保存旧实现和数值基准；不得执行冻结源码做隐性兼容。
3. 运行 python -B -m pytest -q -p no:cacheprovider。
4. 运行 python -B run.py check。
5. 按修改范围测试实际入口和输出，不只测试导入。
6. 输入或数值流程迁移另做前后对照；随机数检查用相同种子和少量Toy，默认单进程。
7. 填写 VALIDATION.md；没有运行的项目明确写“未运行”，不引用旧记录冒充当前验证。
8. 清理本次临时目录。无法读取/清理的既有锁定缓存列明，不盲目修改权限。
9. 更新上述对应说明；不得留“稍后补文档”。

## 文档不应如何写

- AGENT.md 不写当前104/208bin、进程数实测速度、某轮排除线位置等会变的结论。
- README.md 不保存冗长历史对话；只给当前可运行入口与必要科学边界。
- ARCHITECTURE.md 不把“现在实现了”写成“未来计划”。
- frozen 内文档保持原样，是历史证据，不能批量改成新命令。
- docs/VALIDATION.md 不把单元测试通过等同于论文复现或所有参数点正确。
