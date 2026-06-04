import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from numpy.typing import NDArray


def forcedwaves2D(
    force_fn: Callable[[float], float],
    n: int,
    T: float,
) -> NDArray[np.float64]:
    """
    Simulate a 2D Chladni plate using the finite difference time domain (FDTD)
    method and return a contour plot of nodal lines at time T.

    Parameters
    ----------
    force_fn : Callable[[float], float]
        Forcing function applied at the plate centre. Receives the current time
        t (float) and returns the force value (float).
        Example: lambda t: 5 * np.cos(np.pi * np.sqrt(5) * t)

    n : int
        Number of grid intervals along each axis (default 100).
        The grid has (n+1) x (n+1) points over [-1, 1] x [-1, 1].

    T : float
        Total simulation time (default 100.0).

    Returns
    -------
    numpy.NDArray[np.float64]
        A 2D array that contains the vertical displacements of the wave in corresponding coordinates
    """
    h = 2.0 / n
    dt = h / 2.0

    Uo = np.zeros((n + 1, n + 1))  # U at current time step
    Uoo = np.zeros((n + 1, n + 1))  # U at previous time step

    kk = n // 2  # Centre point

    # Precompute reflected boundary indices (Neumann BC)
    idx = np.arange(n + 1)
    im_idx = np.where(idx == 0, 1, idx - 1)  # left/bottom reflect
    ip_idx = np.where(idx == n, n - 1, idx + 1)  # right/top reflect
    # Same arrays work for both row and column directions

    total_steps = int(T / dt)

    # Could use recurrsion here but can't be bothered to deal with recurrsion limits
    for t_step in range(1, total_steps + 2):
        # Discretised Laplacian (5-point stencil)
        laplacian = (
            Uo[im_idx, :] + Uo[ip_idx, :] + Uo[:, im_idx] + Uo[:, ip_idx] - 4.0 * Uo
        ) / h**2

        U = dt**2 * laplacian - Uoo + 2.0 * Uo

        # Centre point with external forcing
        lap_centre = (
                Uo[kk - 3:kk+2, kk-2:kk+3]
                + Uo[kk-1:kk+4, kk-2:kk+3]
                + Uo[kk-2:kk+3, kk-3:kk+2]
                + Uo[kk-2:kk+3, kk-1:kk+4]
                - 4.0 * Uo[kk-2:kk+3, kk-2:kk+3]
        ) / h**2

        t_current = dt * (t_step - 1)

        U[kk-2:kk+3, kk-2:kk+3] = (
                (lap_centre + force_fn(t_current)) * dt**2 - Uoo[kk-2:kk+3, kk-2:kk+3] + 2.0 * Uo[kk-2:kk+3, kk-2:kk+3]
        )

        # Progress in Time
        Uoo = Uo
        Uo = U

    return Uo


def SHM(alpha, omega, t):
    return alpha * np.cos(omega * t)


if __name__ == "__main__":
    from functools import partial
    from display import Display

    N = 100
    T = 100
    w = np.pi * np.sqrt(1**2 + 4**2)
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
