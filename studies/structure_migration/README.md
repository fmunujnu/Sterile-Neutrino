# 四块结构迁移记录

此目录不参与分析。migrate_layout.py、finish_grouping.py、
separate_spectrum_payloads.py 是一次性迁移脚本，不要重复执行。
update_output_paths.py、finish_output_paths.py、move_outputs.ps1是随后输出布局迁移的记录，也不要重跑。
module_map.json 记录旧源码到新模块的对应。

冻结原件：frozen/before_four_block_layout_20260903。
check_parity.py 只执行当前活动代码；旧实现不导入、不执行。
迁移前已记录可见JSON基准，迁移后运行以下命令逐项比较：

```powershell
python -B studies/structure_migration/check_parity.py --baseline outputs/studies/structure_migration/legacy/results/before.json --output-directory outputs/studies/structure_migration/new_check/results
```

不可覆盖的旧基准：outputs/studies/structure_migration/legacy/results/before.json。
新报告显式指定独立输出目录，不能再次覆盖旧批次的报告。
覆盖科学输入哈希、两模型概率/预测、BNB/联合协方差、Fig3a/Fig3b profile、
解析CLs、8 Toy/假设的原始统计量和尾计数。完全相同不代表全参数空间已验证。

tests/test_four_block_layout.py 另外读取冻结源码作AST比较，证明选定数值函数体未改。
这些测试不执行冻结文件。独立prefit函数及原prefit测试已在快照归档。
以后科学实现明确授权改变时，应建立新基准并说明为什么旧迁移基准不再适用，
不能偷偷改写本次before.json使测试通过。
