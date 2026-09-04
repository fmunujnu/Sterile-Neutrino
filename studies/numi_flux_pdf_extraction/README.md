# NuMI PDF 矢量提取研究

这是独立来源研究，不是活动扫描的一环。完整四页提取是当前保留的复现路径；
早期两页脚本、MINERvA基准和布局检查保留作历史验证，不需要每天重跑。
相关数组已经复制进 data/experiments/microboone/numi/inputs/flux_components；
主扫描只读已登记输入，不导入此研究。

## 完整四页流程

入口：

```powershell
powershell -ExecutionPolicy Bypass -File studies/numi_flux_pdf_extraction/run_pages_4_7.ps1 -Python python
```

读取本目录 data/microboone_note_1129/MICROBOONE-NOTE-1129-PUB.pdf，
使用图1--4的蓝色New Flux路径。PDF第4--5页为FHC/RHC的numu、nue，
第6--7页为numu+numubar、nue+nuebar；同模式、同能量格的总数减中微子得到反中微子。

- src/pdf_extract.py：路径选择、坐标轴与刻度识别、线性/log10坐标转换、水平阶梯恢复。
- src/geometry.py：坐标拟合和裁剪判断。
- postprocess_pages_4_7.py：按原阶梯展开到目标bin及相减。
- extract_microboone_pages_4_7.py：四页编排、审计和数组输出。
- render_and_plot_pages_4_7.py：渲染原PDF四页，并单独重绘已经登记的8份CSV。
- tests：此研究自己的测试，不在默认根目录pytest发现范围内。

输出在 outputs/studies/numi_flux_pdf_extraction/<UTC批次>/results/microboone_pages_4_7。
flux_arrays 保存8份CSV；source_coordinate_audit保存绝对PDF坐标与物理坐标审计。
每份数组覆盖0--5 GeV，50个0.1 GeV bin；原图2 GeV以上阶梯更宽，按显示值分段常数展开，
不插值发明未展示的细结构。坐标边界裁剪标记仍保留。

## 验证与早期工具

MINERvA用于有官方TXT真值的提取基准，绝不作为MicroBooNE束流替代：
python studies/numi_flux_pdf_extraction/run_benchmark.py。
输出位于 outputs/studies/numi_flux_pdf_extraction/<UTC批次>/results/minerva_numu_fhc，
包含 recovered_flux.csv、validation.json、真值/恢复值投影叠图与识别对象记录。

check_microboone_note.py 检查MicroBooNE实际图形/坐标轴识别。
extract_microboone_pages_5_6.py、run_pages_5_6.ps1是原两页版本，非完整八数组流程。

工具使用本地Python、pdfplumber、NumPy、Pillow、Poppler；下载原件后提取不依赖模型API、
云服务或网络。依赖清单见本目录requirements.txt；机器上的Python/Poppler路径须自行确认。

## 解释边界和维护

图上重合仅验证显示路径恢复，不证明等于未发布的完整ROOT直方图。
改提取、分bin或相减算法时，更新本说明、来源/审计记录，并运行此研究测试和真值闭合；
若要更新正式data输入，还须遵守 docs/MAINTENANCE.md，不自动覆盖已登记数组。
