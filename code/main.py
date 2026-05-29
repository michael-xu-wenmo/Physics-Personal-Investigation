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


def freewave():
    N = 50
    k = 30
    value, func = eigenwaves2D(N, k, bound="Neumann")
    svalue, sfunc = superposition(value, func)
    d = Display(svalue, sfunc, "Chladni")
    d.export(d.plot_energy(), "freq.png")
    for i, fig in enumerate(d.plot_wavefunction()):
        d.export(fig, f"pattern{i}.png")
        plt.close()


def forcedwave():
    N = 100
    T = 100
    w = np.pi * np.sqrt(1**2 + 2**2)
    alpha = 5.0

    U = forcedwaves2D(
        force_fn=partial(SHM, alpha, w),
        n=N,
        T=T,
    )

    display = Display([w], [U], "Forced Wave")
    for i, fig in enumerate(display.plot_wavefunction()):
        display.export(fig, f"pattern{i}.png")
        plt.close()


if __name__ == "__main__":
    start = time()
    forcedwave()
    end = time()
    print(end - start)  # benchmark
