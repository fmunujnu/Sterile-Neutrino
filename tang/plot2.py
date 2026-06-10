import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import sys
import io

# 强制改变标准输出流的编码为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 1. 手动输入区域 (请在此处修改你的理论参数)
# ==========================================
A_manual = 216480      # 手动输入旋光常数 A
lambda_0_manual = 126.9  # 手动输入特征波长 \lambda_0 (单位: nm)

print("="*50)
print(f"当前手动设定的参数：")
print(f"A        = {A_manual}")
print(f"λ0       = {lambda_0_manual} nm")
print("="*50)


# ==========================================
# 2. 硬编码实验数据
# ==========================================
lambda_c_data = np.array([450, 500, 530, 570, 600, 635], dtype=float)
delta_lambda_data = np.array([139, 184, 223, 261, 317, 409], dtype=float)

# 已知实验常量
c = 0.5  # 浓度 g/ml
L = 4.0  # 长度 dm


# ==========================================
# 3. 定义理论公式函数
# ==========================================
def theoretical_model(x, A, lambda_0):
    # 公式: \Delta \lambda = (\pi * (x^2 - lambda_0^2)^2) / (4 * A * c * L * x)
    numerator = np.pi * (x**2 - lambda_0**2)**2
    denominator = 4 * A * c * L * x
    return numerator / denominator


# ==========================================
# 4. 计算基于手动参数的预测值与统计量
# ==========================================
# 计算对应实验波长下的理论预测值
delta_lambda_pred = theoretical_model(lambda_c_data, A_manual, lambda_0_manual)

# 计算统计量
r2 = r2_score(delta_lambda_data, delta_lambda_pred)
rmse = np.sqrt(np.mean((delta_lambda_data - delta_lambda_pred)**2))

print("比对统计结果:")
print(f"决定系数 (R² Score)     : {r2:.6f}  (越接近 1 说明手动参数越贴合实际)")
print(f"均方根误差 (RMSE)       : {rmse:.4f} nm")
print("="*50)


# ==========================================
# 5. 联合比对图绘制
# ==========================================
plt.figure(figsize=(9, 6), dpi=100)

# 绘制实验散点图
plt.scatter(lambda_c_data, delta_lambda_data, color='red', s=80, zorder=5, label='Experimental Data')

# 绘制手动参数下的理论连续曲线
x_fine = np.linspace(400, 700, 500)
y_fine = theoretical_model(x_fine, A_manual, lambda_0_manual)
plt.plot(x_fine, y_fine, color='green', linewidth=2, linestyle='-', label='Manual Theoretical Curve')

# 界面美化与标注
plt.title(r'$\delta\lambda$ vs $\lambda_c$ Manual Comparison', fontsize=14, pad=15)
plt.xlabel(r'Wavelength $\lambda_c$ (nm)', fontsize=12)
plt.ylabel(r'Bandwidth $\delta\lambda$ (nm)', fontsize=12)

# 在图表上展示手动输入的参数和当前的 R²
text_box = (f"Manual Settings:\n"
            f"A = {A_manual}\n"
            f"$\lambda_0$ = {lambda_0_manual} nm\n"
            f"Current $R^2$ = {r2:.5f}")
plt.text(420, max(delta_lambda_data)*0.7, text_box, fontsize=11, 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='lower right')

# 显示图形
plt.show()