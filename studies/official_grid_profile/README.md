# 官方网格研究

读取本地官方三维delta-chi2 TXT，对照profile和原阈值画法。
该文件是NuMI命名的官方发布网格，不因此成为BNB+NuMI联合网格。
不进入本地预测、协方差、解析CLs或Toy CLs计算。

入口：python studies/official_grid_profile/profile_official_grid.py。
输出：outputs/studies/official_grid_wilks/<UTC批次>/results。大型TXT位于data共享层并被Git忽略。

此工具保留原统计方法，不能把其Wilks对照称为本项目CLs结果。
输入版本、参数坐标或阈值改变时更新本说明、来源记录和结果metadata。
