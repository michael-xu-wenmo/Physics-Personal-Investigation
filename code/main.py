# Scientific Modules
import numpy as np
import matplotlib.pyplot as plt

# Models
from eigenwave import eigenwaves2D
from forcedwave import forcedwaves2D, SHM

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
    N = 50
    k = 30

    value, func = eigenwaves2D(N, k, bound="Neumann")
    value, func = superposition(value, func)
    beta_sq = value / L**2  # scale to physical domain
    frequencies = (
        beta_sq * c / (2 * np.pi)
    )  # attempt to obtain the correct frequency. Still far off :(

    d = Display(frequencies, func, "Wave")
    d.export(d.plot_energy(), "freq.png")
    for i, fig in enumerate(d.plot_wavefunction()):
        d.export(fig, f"pattern{i}.png")
        plt.close()


def forcedwave():
    N = 100
    T = 100
    alpha = 5.0

    freqs = []
    waves = []
    for i in range(0, 5):
        for j in range(0, 5):
            w = np.pi * np.sqrt(i**2 + j**2)
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


if __name__ == "__main__":
    start = time()
    # freewave()
    forcedwave()
    end = time()
    print(end - start)  # benchmark
