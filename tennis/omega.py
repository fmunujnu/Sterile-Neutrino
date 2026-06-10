import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 保持与你论文相同的字体和美化风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Noto Serif SC', 'SimSun', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义刚体参数 (中间轴不稳定配置)
I1, I2, I3 = 5.0, 4.0, 1.0  # I1 > I2 > I3
omega_0 = 1.0
eps = 1e-4  # 极小的初始扰动

# 2. 刚体欧拉动力学方程
def euler_equations(t, w):
    w1, w2, w3 = w
    dw1 = ((I2 - I3) / I1) * w2 * w3
    dw2 = ((I3 - I1) / I2) * w3 * w1
    dw3 = ((I1 - I2) / I3) * w1 * w2
    return [dw1, dw2, dw3]

# 3. 设置仿真时间 (根据你之前的周期公式，给足至少2个周期的时间)
t_span = (0, 150)
t_eval = np.linspace(t_span[0], t_span[1], 2000)
initial_state = [eps, omega_0, eps] # 在1轴和3轴施加轻微扰动

# 数值积分求解
sol = solve_ivp(euler_equations, t_span, initial_state, t_eval=t_eval, rtol=1e-9, atol=1e-9)

# 4. 绘制经典的时历演化曲线
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

ax.plot(sol.t, sol.y[0], color='#D95319', linewidth=1.5, label=r'$\omega_1$')
ax.plot(sol.t, sol.y[1], color='#333333', linewidth=2.0, label=r'$\omega_2$')
ax.plot(sol.t, sol.y[2], color='#1f77b4', linewidth=1.5, label=r'$\omega_3$')

ax.set_xlabel('时间 Time $t$ (s)', fontsize=11)
ax.set_ylabel('角速度 $\omega$ (rad/s)', fontsize=11)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper right', frameon=True)

# 标注物理现象
# 借助数值结果找翻转点（当w2第一次穿过0时）

plt.title('三轴角速度随时间变化', fontsize=12, pad=12)
plt.tight_layout()

plt.savefig('tennis_racket_time_history.png', bbox_inches='tight')
plt.show()