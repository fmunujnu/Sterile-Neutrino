import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# =====================================================
# Parameters
# =====================================================

I1 = 1.0
I2 = 2.0
I3 = 3.0

E = 3.0

# =====================================================
# Energy ellipsoid
#
# E = 1/2(I1w1²+I2w2²+I3w3²)
# =====================================================

a = np.sqrt(2*E/I1)
b = np.sqrt(2*E/I2)
c = np.sqrt(2*E/I3)

# angular coordinates

u = np.linspace(0, 2*np.pi, 300)
v = np.linspace(0, np.pi, 150)

U, V = np.meshgrid(u, v)

W1 = a*np.cos(U)*np.sin(V)
W2 = b*np.sin(U)*np.sin(V)
W3 = c*np.cos(V)

# =====================================================
# Angular momentum on ellipsoid
#
# L²=(I1w1)²+(I2w2)²+(I3w3)²
# =====================================================

L2 = (
    (I1*W1)**2
    + (I2*W2)**2
    + (I3*W3)**2
)

# =====================================================
# Figure
# =====================================================

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(
    111,
    projection='3d'
)

# =====================================================
# Surface
# =====================================================

ax.plot_surface(
    W1,
    W2,
    W3,
    color='lightgray',
    edgecolor='none',
    alpha=0.65,
    shade=True
)

# =====================================================
# Contours of L² on ellipsoid
# =====================================================

levels = np.linspace(
    L2.min(),
    L2.max(),
    18
)

ax.contour(
    W1,
    W2,
    W3,
    L2,
    levels=levels,
    colors='0.35',
    linewidths=0.8
)

# =====================================================
# Principal axes
# =====================================================

axis_scale = 1.35*max(a,b,c)

# omega1

ax.quiver(
    -axis_scale,0,0,
    2*axis_scale,0,0,
    color='black',
    arrow_length_ratio=0.04,
    linewidth=1.5
)

# omega2

ax.quiver(
    0,-axis_scale,0,
    0,2*axis_scale,0,
    color='black',
    arrow_length_ratio=0.04,
    linewidth=1.5
)

# omega3

ax.quiver(
    0,0,-axis_scale,
    0,0,2*axis_scale,
    color='black',
    arrow_length_ratio=0.04,
    linewidth=1.5
)

# =====================================================
# Labels
# =====================================================

ax.text(
    axis_scale*1.05,
    0,
    0,
    r'$\omega_1$',
    fontsize=18
)

ax.text(
    0,
    axis_scale*1.05,
    0,
    r'$\omega_2$',
    fontsize=18
)

ax.text(
    0,
    0,
    axis_scale*1.05,
    r'$\omega_3$',
    fontsize=18
)

# =====================================================
# Center marker
# =====================================================

ax.scatter(
    [0],
    [0],
    [0],
    color='black',
    s=20
)

# =====================================================
# View angle
# =====================================================

ax.view_init(
    elev=22,
    azim=-55
)

# =====================================================
# Appearance
# =====================================================

ax.set_box_aspect([1,1,1])

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_axis_off()

# =====================================================
# Save high-DPI PNG
# =====================================================

plt.savefig(
    "Poinsot_Arnold_Style.png",
    dpi=1200,
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=False
)

plt.close()

print("Saved: Poinsot_Arnold_Style.png")