import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import read_data as rd

# 输入数据
observed = np.array([83, 90, 63, 58, 49, 45, 35, 67], dtype=float)
background = np.array([70.91, 81.51, 61.94, 58.95, 50.01, 44.49, 33.73, 65.27], dtype=float)

test_file = r"E:\sterile neutrino data\HEPData-ins3088922-v1-Unconstrained_14_channels.csv"

data, bkg, sigbkg = rd.read_three_spectra(test_file)

test_file2 = r"E:\sterile neutrino data\HEPData-ins3088922-v1-14_channel_covariance_matrix.csv"  # 改成你的文件路径

cov_raw = rd.read_covariance_matrix(
    test_file2,
    make_symmetric=False          # 如果只给了半个矩阵就设为True
)

background = bkg


# 转换为 NumPy 数组
cov_raw = np.array(cov_raw, dtype=float)

# 计算绝对协方差矩阵
cov_abs = np.outer(background, background) * cov_raw

# 从绝对协方差计算相关系数矩阵
std = np.sqrt(np.diag(cov_abs))
std_outer = np.outer(std, std)
corr = cov_abs / std_outer

# 设置对称对数归一化的阈值（根据数据范围调整）
linthresh = 1.0   # 在此阈值内的值使用线性映射，之外使用对数

# 创建两个子图并排显示
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ---- 左图：绝对协方差（对称对数色标） ----
norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=cov_abs.min(), vmax=cov_abs.max())
im1 = ax1.imshow(cov_abs, cmap='coolwarm', aspect='auto', origin='upper', norm=norm)
ax1.set_xticks(range(8))
ax1.set_yticks(range(8))
ax1.set_xticklabels(range(1, 9))
ax1.set_yticklabels(range(1, 9))
ax1.set_xlabel('Index')
ax1.set_ylabel('Index')
ax1.set_title('Absolute Covariance (SymLog Norm)')
cbar1 = fig.colorbar(im1, ax=ax1, label='Covariance (log scale)')

# ---- 右图：相关系数矩阵（线性色标） ----
im2 = ax2.imshow(corr, cmap='coolwarm', aspect='auto', origin='upper', vmin=-1, vmax=1)
ax2.set_xticks(range(8))
ax2.set_yticks(range(8))
ax2.set_xticklabels(range(1, 9))
ax2.set_yticklabels(range(1, 9))
ax2.set_xlabel('Index')
ax2.set_ylabel('Index')
ax2.set_title('Correlation Coefficient Matrix')
cbar2 = fig.colorbar(im2, ax=ax2, label='Correlation')

plt.tight_layout()
plt.show()

print("Correlation coefficient matrix:")
print(corr)