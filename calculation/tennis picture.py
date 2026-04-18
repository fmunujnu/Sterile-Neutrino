import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

# ==================== 固定参数 ====================
I1 = 5.0
I3 = 2.0

omega1 = 0.1
omega2 = 100.0
omega3 = 2.5
# ================================================

# 扫描 I2
I2_list = np.linspace(1,3,2000)

I2_vals = []
T_vals = []
sep_vals = []

for I2 in I2_list:

    # 能量
    E = 0.5*(I1*omega1**2 + I2*omega2**2 + I3*omega3**2)

    # 角动量平方
    L2 = (I1*omega1)**2 + (I2*omega2)**2 + (I3*omega3)**2

    # separatrix 指标
    sep = L2 - 2*E*I2

    # k^2
    numerator = (I3-I2)*(L2-2*E*I1)
    denominator = (I2-I1)*(2*E*I3-L2)

    if abs(denominator) < 1e-12:
        continue

    k2 = numerator/denominator

    # 椭圆积分
    K = ellipk(k2)

    denom1 = (I1-I2)*(I1-I3)

    term1 = np.sqrt(I2*I3/denom1)

    denom2 = -(2*E*I2 - L2)

    term2 = np.sqrt(I1*(I1-I2)/denom2)

    T = 4*K*term1*term2

    if np.isfinite(T):

        I2_vals.append(I2)
        T_vals.append(T)
        sep_vals.append(sep)

# ==================== 图1 ====================

plt.figure(figsize=(6,5))
plt.plot(I2_vals,T_vals)
plt.xlim(2.0, 3.0)
plt.xlabel("I2")
plt.ylabel("Period T")
plt.title("Period vs I2")

plt.grid(True)
plt.show()

# ==================== 图2 ====================

plt.figure(figsize=(6,5))
plt.plot(I2_vals,T_vals)
plt.xlim(2.0, 3.0)
plt.yscale("log")

plt.xlabel("I2")
plt.ylabel("Period T (log scale)")
plt.title("Period vs I2")

plt.grid(True)
plt.show()

# ==================== 图3 ====================

plt.figure(figsize=(6,5))
plt.plot(I2_vals,sep_vals)
plt.xlim(2.0, 3.0)
plt.axhline(0)

plt.xlabel("I2")
plt.ylabel("L^2 - 2 E I2")
plt.title("Separatrix Condition")

plt.grid(True)
plt.show()