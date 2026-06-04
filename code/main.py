# Scientific Modules
import numpy as np
import matplotlib.pyplot as plt

# Models
from eigenwave import eigenwaves2D
from forcedwave import forcedwaves2D, SHM
from plate import solve_kirchhoff_love

# Utility Imports
from time import time
from display import Display
from functools import partial
from superposition import superposition

# Physical parameters
E = 70e9  # Pa
nu = 0.33
rho = 2700  # kg/m³
h = 0.002  # m (2 mm)
L = 0.25  # m (25 cm)

# Flexural rigidity
D = E * h**3 / (12 * (1 - nu**2))

# wave speed factor
c = np.sqrt(D / (rho * h))


def freewave():
    N = 100
    k = 200

    value, func = eigenwaves2D(N, k, bound="Neumann")
    value, func = superposition(value, func)
    beta_sq = value / L**2  # scale to physical domain
    frequencies = (
        beta_sq * c / (2 * np.pi)
    )  # attempt to obtain the correct frequency. Still far off :(

    d = Display(value, func, "Wave")
    d.export(d.plot_energy(), "freq.png")
    for i, fig in enumerate(d.plot_wavefunction()):
        d.export(fig, f"pattern{i}.png")
        plt.close()


def forcedwave():
    N = 100
    T = 100
    alpha = 1000.0

    freqs = []
    waves = []
    for i in range(1):
        for j in range(1):
            w = np.pi * np.sqrt(3**2 + 5**2)
            U = forcedwaves2D(
                force_fn=partial(SHM, alpha, w),
                n=N,
                T=T,
            )
            freqs.append(w)
            waves.append(U)

    display = Display(freqs, waves, "Forced Wave")
    for i, fig in enumerate(display.plot_wavefunction()):
        display.export(fig, f"pattern{i}.png")
        plt.close()

def platewave():
    L       = 0.24 # m
    E       = 69e9          # Pa
    h_plate = 0.001         # m
    nu      = 0.33
    rho     = 2700.0        # kg/m³
    rho_h   = rho * h_plate # surface mass density  [kg/m²]
    D       = E * h_plate**3 / (12.0 * (1.0 - nu**2))

    # First resonant frequency (from paper, Figure 12): f1 ≈ 609.7 Hz
    for i in range(1):
        f1  = 995.2
        xi  = 2.0 * np.pi * f1  # angular frequency
        F0  = 1e10              # large amplitude to quickly excite the mode

        # Forcing patch: 2 cm x 2 cm square centred on plate
        xc, yc = L / 2.0, L / 2.0

        def force_fn(X, Y, t):
            mask = (np.abs(X - xc) <= 0.01) & (np.abs(Y - yc) <= 0.01)
            return np.where(mask, F0 * np.cos(xi * t), 0.0)

        N = 160
        ts,ws = solve_kirchhoff_love(
            force_fn = force_fn,
            L        = L,
            N        = N,
            T        = 10,
            rho_h    = rho_h,
            E        = E,
            h_plate  = h_plate,
            nu       = nu,
            save_path= "kirchhoff_love_nodal_lines.png",
        )

        fs = np.array([f1 for _ in range(len(ws))])

        d = Display(fs,ws,"Plates")
        for i, fig in enumerate(d.plot_wavefunction()):
            d.export(fig,f"plate{i}.png")
            plt.close()

if __name__ == "__main__":
    start = time()
    freewave()
    #forcedwave()
    #platewave()
    end = time()
    print(end - start)  # benchmark
