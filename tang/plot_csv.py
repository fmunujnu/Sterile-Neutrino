import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import sys
import io

# 强制改变标准输出流的编码为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# 1. 硬编码表格中的实验数据
# 考虑到两组重复实验，我们把所有数据点都引入
lambda_c_data = np.array([450, 500, 530, 570, 600, 635], dtype=float)
delta_lambda_data = np.array([91, 125, 139, 184, 223, 261], dtype=float)

# 已知实验常量
c = 0.5  # g/ml
L = 4.0  # dm

# 2. 定义理论公式模型
# 传入参数: x (即 lambda_c), A, lambda_0
def theoretical_model(x, A, lambda_0):
    # 限制 lambda_0 的合理物理范围，避免分母或开方出现负数/零
    # 公式: delta_lambda = (pi * (x^2 - lambda_0^2)^2) / (4 * A * c * L * x)
    numerator = np.pi * (x**2 - lambda_0**2)**2
    denominator = 4 * A * c * L * x
    return numerator / denominator

# 3. 使用非线性最小二乘法进行曲线拟合
# 给出 A 和 lambda_0 的初始猜测值 (例如 A=300, lambda_0=150)
initial_guess = [300.0, 150.0]
# 设定参数边界：A > 0, 0 < lambda_0 < 400 (通常在紫外区)
bounds = ((0.1, 10.0), (1000.0, 40000.0)) # 也可以不设，这里让 scipy 自由寻找

popt, pcov = curve_fit(theoretical_model, lambda_c_data, delta_lambda_data, p0=initial_guess)
A_opt, lambda_0_opt = popt

print("="*50)
print("拟合计算结果:")
print(f"拟合得到的参数 A        : {A_opt:.4f}")
print(f"拟合得到的特征波长 λ0    : {lambda_0_opt:.2f} nm")

# 4. 计算统计量 (R² 和 RMSE)
# 计算预测值
delta_lambda_pred = theoretical_model(lambda_c_data, A_opt, lambda_0_opt)
r2 = r2_score(delta_lambda_data, delta_lambda_pred)
rmse = np.sqrt(np.mean((delta_lambda_data - delta_lambda_pred)**2))

print(f"决定系数 (R² Score)     : {r2:.6f}")
print(f"均方根误差 (RMSE)       : {rmse:.4f}")
print("="*50)

# 5. 绘制联合比对图
plt.figure(figsize=(9, 6), dpi=100)

# 绘制实验散点图
plt.scatter(lambda_c_data, delta_lambda_data, color='red', s=80, zorder=5, label='Experimental Data (Table)')

# 绘制理论连续曲线
x_fine = np.linspace(400, 700, 500)
y_fine = theoretical_model(x_fine, A_opt, lambda_0_opt)
plt.plot(x_fine, y_fine, color='blue', linewidth=2, label=f'Theoretical Curve (Fit)')

# 界面美化与标注
plt.title(r'$\delta\lambda$ vs $\lambda_c$ Comparison & Fitting', fontsize=14, pad=15)
plt.xlabel(r'Wavelength $\lambda_c$ (nm)', fontsize=12)
plt.ylabel(r'Bandwidth $\delta\lambda$ (nm)', fontsize=12)

# 在图表上展示拟合结果和 R²
text_box = (f"Fit parameters:\n"
            f"A = {A_opt:.2f}\n"
            rf"$\lambda_0$ = {lambda_0_opt:.1f} nm" + "\n"
            rf"$R^2$ = {r2:.5f}")
plt.text(420, 200, text_box, fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.savefig('fit_comparison.png', dpi=300, bbox_inches='tight')
print("拟合对比图已保存为: fit_comparison.png")