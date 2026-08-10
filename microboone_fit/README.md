# MicroBooNE BNB reference-reweight analysis

本项目使用 MicroBooNE 公开 BNB 前四个 CC 通道，在一个明确声明的 3+1 参考点上构造可审计的经验重加权核。它不寻找或声称恢复合作组原始 MC；所有活跃数值输入均为 CSV/YAML，所有元数据与结果均为 JSON/CSV。

## 最常用的三个命令

首次安装：

```powershell
cd "E:\Sterile Neutrino\microboone_fit"
python -m pip install -e ".[dev]"
```

完整检查：

```powershell
python scripts/check.py
```

重画 HEPData 公开 BNB 四通道图：

```powershell
python scripts/plot_published_bnb.py --output runs/reproduction/published_bnb_four_channels.png
```

运行精确 `sin²(2θμe)-Δm²41` profile：

```powershell
python scripts/scan.py --mode appearance-profile
```

扫描结果写入 `runs/<时间>_appearance-profile/result.csv`，运行说明写入同目录的 `metadata.json`。
默认使用两个方向各 61 点的对数网格（共 3721 个 profile 点）。可用 `--grid-points 81` 做更高分辨率收敛检查，或用 `--grid-points 41` 较快测试；显式 grid 参数会覆盖默认坐标。少于 8×8 点时程序拒绝画容易误导的等高线。

## 可调扫描参数

```powershell
python scripts/scan.py `
  --mode appearance-profile `
  --delta-m2-grid-eV2 0.1,0.3,1.0,3.0,10.0 `
  --sin2-2theta-mue-grid 0.0001,0.001,0.01,0.1
```

其他模式：

- `--mode prefit`：同时拟合 `delta_m2_41_eV2`、`sin2_theta14`、`sin2_theta24`；
- `--mode s14-profile`：固定 `(delta_m2_41_eV2, sin2_theta14)`，profile `sin2_theta24`；
- `--mode appearance-profile`：固定 `(delta_m2_41_eV2, sin2_2theta_mue)`，在精确物理约束下 profile `sin2_theta14` 并推导 `sin2_theta24`。

零惰性混合锚点与基线在 `configs/bnb_3nu_anchor.yaml` 中声明。论文图 1 的示意点也只作为注释保留，绝不参与锚定。参数只使用：

```text
delta_m2_41_eV2    Δm²41，单位 eV²
sin2_theta14       sin²θ14
sin2_theta24       sin²θ24
```

BNB 基线也在同一配置中显式设为 `baseline_km: 0.4685`。这是论文给出的 BNB 靶到 MicroBooNE 探测器距离；当前 kernel 使用单基线近似，不冒充合作组逐事件产生点分布。

## 简化调用链

```text
scripts/check.py                 一键检查全部输入与闭合
scripts/build_anchor.py          公开预测 + 背景 + 束流 + Reco → 文本 kernel
scripts/scan.py                  kernel + 数据 + 协方差 + 3+1 概率 → profile CSV
```

内部核心按职责保留：

```text
parameters.py                    参数定义
models/three_plus_one.py         论文短基线极限下的精确 3+1 概率
templates.py                     CSV/JSON kernel 读取与预测
covariance.py                    CSV/JSON 协方差读取及 Cholesky χ²
fitting.py                       prefit 与 profile
```

## HEPData 三个数据块

`HEPData-ins3088922-v1-Unconstrained_14_channels.csv` 中：

```text
Data                  只进入 χ²
Background            在参考重加权中冻结并只加一次
Signal + Background   参考总预测
Signal                 由 (Signal + Background) - Background 得到
```

`plot_published_bnb.py` 直接画 `Background` 和 `Signal + Background` 两条线，没有执行 `total + background`。

## 60 true bins 与 26 reco bins

不把 26-bin reco 谱插值成 60-bin true 谱，也不压缩 true-energy 轴。项目保留：

```text
60 true-energy bins：0--3 GeV，每 bin 0.05 GeV
26 reco bins：25 个 0.1 GeV bin + 2.5--3.0 GeV overflow
Reco 矩阵：26 x 60，R(reco|true)
```

`build_anchor.py` 在每个公开 reco bin 中，用 `Reco × 束流 × 参考概率` 分配 true-energy 权重，并用公开 Signal 强制归一化。因此参考点逐 bin 精确闭合，又不把 reco energy 冒充 true energy。

## 可见数据位置

```text
data/raw/                 HEPData CSV/YAML 原始输入
data/inputs/              可见 BNB flux CSV 与来源 JSON
data/derived/             可见 Reco CSV、协方差 CSV/JSON
data/anchor/              八个过程 kernel CSV、背景 CSV、闭合 CSV、metadata JSON
```

活跃分析不读取 NPZ、pickle 或其他二进制数值容器。PNG 只作为可见图片输出，不作为拟合输入。

## 科学边界

当前结果是“BNB 四通道、公开零惰性混合预测锚定、公开背景类别冻结”的经验重加权分析。metadata 明确记录了以下假设：未知截面与效率不能从一个 reco 谱唯一反演，而是假定其参考加权效应可由 `Reco×flux` 真能量形状先验和每个 reco bin 的一个比例因子吸收；同一选择通道内中微子和反中微子共享该有效比例；2022 Reco 作为 true-energy 分配先验。

配置注释中的 `1.2 eV², sin²θ24=0.018, sin²(2θμe)=0.003` 是论文图 1 的示意点，不是论文 best-fit，也不是 kernel 锚点。

这里的“固定背景”只表示 HEPData `Background` 类别在本近似中保持不变。公开表没有给出该类别内可振荡束流中微子、宇宙线、探测器外事件等逐成分真能量模板，因此这是当前最重要的不可消除限制。协方差在每个扫描点按当前/名义预测比保持分数系统误差，并重新加入当前预测的 Pearson 统计项；由于公开协方差也没有逐背景成分拆分，这仍是公开数据近似。论文正式限制还使用全部 14 通道和伪实验 `CLs`；本项目输出是 BNB-only 的诊断性 profile `Δχ²`，不能标成论文 95% 排除线。

profile 会搜索完整 `0 <= sin²θ14 <= 1` 与 `0 <= sin²θ24 <= 1` 域；固定外观振幅时不会丢弃大 `sin²θ14` 支。prefit 还显式检查 `sin²θ24=0` 和 `sin²θ14=0` 两个零外观边界，避免连续优化器漏掉边界极小值。

未来 1+3+1 只替换概率模型，必须在第二惰性态混合为零时逐点回到 3+1；数据、kernel、背景、Reco、协方差与 χ²不得复制或改变。
