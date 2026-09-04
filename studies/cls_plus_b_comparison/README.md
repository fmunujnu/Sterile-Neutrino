# 历史CLs+b对照

独立读取已有Run1解析结果，对比CLs=p4/p3和CLs+b=p4的0.05轮廓。
不生成Toy、不重新profile，也不改变当前活动的CLs方法。

入口：python studies/cls_plus_b_comparison/plot_cls_plus_b.py。
必须用 --fig3a-result 和 --fig3b-result 显式传入已有CSV，不再绑定历史目录。
输出：outputs/studies/cls_plus_b_non_toy_comparison/<UTC批次>/results。

仅为历史研究保留，不从核心调用。若输入列名变化，显式调整此工具与说明；
不要把该方法重新加入默认分析。
