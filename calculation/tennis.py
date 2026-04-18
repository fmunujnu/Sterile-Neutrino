import numpy as np
from scipy.special import ellipk

# ==================== 硬编码输入（请在此修改数值） ====================
I1 = 5.0  # 转动惯量1
I2 = 3.0  # 转动惯量2
I3 = 2.0  # 转动惯量3
omega1 = 0.01  # 角速度分量1
omega2 = 2.0  # 角速度分量2
omega3 = omega1  # 角速度分量3
# =================================================================

# 计算能量 E 和角动量平方 L²
E = 0.5 * (I1 * omega1 ** 2 + I2 * omega2 ** 2 + I3 * omega3 ** 2)
L2 = (I1 * omega1) ** 2 + (I2 * omega2) ** 2 + (I3 * omega3) ** 2

print(f"E = {E}")
print(f"L² = {L2}")

# 计算 k²
numerator_k = (I3 - I2) * (L2 - 2 * E * I1)
denominator_k = (I2 - I1) * (2 * E * I3 - L2)
k2 = numerator_k / denominator_k if denominator_k != 0 else float('nan')
print(f"k² = {k2}")

# 检查条件并计算 T

k = np.sqrt(k2)

# 计算椭圆积分 K(k)
K = ellipk(k)

print(f"K = {K}")

# 计算两个平方根项
term1 = np.sqrt(I2 * I3 / ((I1 - I2) * (I1 - I3)))
term2 = np.sqrt(I1 * (I1 - I2) / - (2 * E * I2 - L2))

# 计算 T
T = 4 * K * term1 * term2
print(f"T = {T}")