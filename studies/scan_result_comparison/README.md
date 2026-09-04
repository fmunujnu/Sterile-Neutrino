# 原完成扫描对照脚本

原一次性对照入口已移入frozen/before_output_layout_20260903，主功能已经归并到 output.py。
日常统一使用 python run.py compare，不再扩展此处的绘图实现。

只读取已完成CSV及metadata，画排除线，不重新预测、profile或计算CLs。
原脚本保留用于追溯，不能被活动代码导入。
活动入口必须显式提供四个结果目录，命令见docs/OUTPUTS.md；方法是否全Toy应由CSV/metadata确认。
变更活动列名或统计标记时同步检查读取兼容性，不能静默混合不同方法。
