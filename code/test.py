import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters
n = 100  # Number of intervals on each axis
alpha = 5
m = 1
mm = 2
T = 100
w = np.pi * np.sqrt(1**2 + 2**2)

h = 2 / n
dt = h / 2

x = np.linspace(-1, 1, n + 1)
y = np.linspace(-1, 1, n + 1)


# Force function
def force(eps, w, t):
    return eps * np.cos(w * t)


# Initial conditions
Uo = np.zeros((n + 1, n + 1))
Uoo = np.zeros((n + 1, n + 1))
U = np.zeros((n + 1, n + 1))

# Index of center point (1-based (n+2)/2 → 0-based n//2)
kk = n // 2  # equivalent to (n+2)/2 - 1 in 0-based indexing

# Set up figures
fig1 = plt.figure(1, figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection="3d")

fig2 = plt.figure(2, figsize=(6, 6))
ax2 = fig2.add_subplot(111)

X, Y = np.meshgrid(x, y, indexing="ij")

total_steps = int(T / dt)

for t_step in range(1, total_steps + 1):
    # --- Vectorised finite-difference update ---
    # Build padded arrays that implement the Neumann-like boundary reflections
    # used in the original MATLAB (i==1 → i_1=2, i==n+1 → i1=n, etc.)

    # Interior + boundary indices with reflection
    im = np.arange(n + 1)  # row indices 0..n
    ip = np.arange(n + 1)
    jm = np.arange(n + 1)
    jp = np.arange(n + 1)

    im_idx = np.where(im == 0, 1, im - 1)  # reflect left boundary
    ip_idx = np.where(im == n, n - 1, im + 1)  # reflect right boundary
    jm_idx = np.where(jm == 0, 1, jm - 1)
    jp_idx = np.where(jp == n, n - 1, jp + 1)

    laplacian = (
        Uo[im_idx, :] + Uo[ip_idx, :] + Uo[:, jm_idx] + Uo[:, jp_idx] - 4 * Uo
    ) / h**2

    U = (dt**2) * laplacian - Uoo + 2 * Uo

    # --- Centre point with forcing ---
    lap_kk = (
        Uo[kk - 1, kk]
        + Uo[kk + 1, kk]
        + Uo[kk, kk - 1]
        + Uo[kk, kk + 1]
        - 4 * Uo[kk, kk]
    ) / h**2

    U[kk, kk] = (
        (lap_kk + force(alpha, w, dt * (t_step - 1))) * dt**2
        - Uoo[kk, kk]
        + 2 * Uo[kk, kk]
    )

    # --- Time-step bookkeeping ---
    Uoo = Uo.copy()
    Uo = U.copy()

    # --- Plot every 10 steps ---
    if t_step % 10 == 0:
        # Surface plot
        ax1.cla()
        ax1.plot_surface(X, Y, U, cmap=cm.viridis, linewidth=0, antialiased=False)
        ax1.set_title(f"Wave at t = {t_step * dt:.2f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("U")

        # Zero-contour plot
        ax2.cla()
        ax2.contour(x, y, U, levels=[0], colors="blue")
        ax2.set_title(f"Zero contour at t = {t_step * dt:.2f}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_aspect("equal")

        plt.pause(0.05)

plt.show()
