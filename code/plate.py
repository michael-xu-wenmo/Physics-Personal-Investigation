"""
kirchhoff_love.py
-----------------
Solves the generalised Kirchhoff-Love plate equation

    rho_h * w_tt = -K0*w + T*∇²w - D*∇⁴w - K1*w_t + T1*∇²w_t + F(x, t)

on the square domain [0, L] x [0, L] using:
  - Second-order centred finite differences for all spatial operators
  - An implicit Newmark-Beta (NB2) time-stepping scheme (β=1/4, γ=1/2),
    which is unconditionally stable — important for ∇⁴ whose eigenvalues
    scale as h⁻⁴ and would make any explicit scheme extremely restrictive.
  - Free boundary conditions at all edges (zero bending moment and shear),
    with ghost-point extrapolation and the corner condition ∂²w/∂x∂y = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import sparse
from scipy.sparse.linalg import splu
from typing import Callable

# Spatial operators

def _apply_free_bc(W: np.ndarray, nu: float) -> np.ndarray:

    # Ghost rows
    #W[0, :]  = 2.0 * W[1, :]  - W[2, :] - nu*(W[1,])
    #W[-1, :] = 2.0 * W[-2, :] - W[-3, :]
    #W[:, 0]  = 2.0 * W[:, 1]  - W[:, 2]
    #W[:, -1] = 2.0 * W[:, -2] - W[:, -3]

    ## Ghost Corners
    #W[0, 0]   = W[2, 0]   + W[0, 2]   - W[2, 2]
    #W[-1, 0]  = W[-3, 0]  + W[-1, 2]  - W[-3, 2]
    #W[0, -1]  = W[2, -1]  + W[0, -3]  - W[2, -3]
    #W[-1, -1] = W[-3, -1] + W[-1, -3] - W[-3, -3]

    W[0, 1:-1] = (2.0*W[1, 1:-1] - W[2, 1:-1]
                  - nu*(W[1, 2:] - 2.0*W[1, 1:-1] + W[1, :-2]))
    # Right ghost (row -1), edge is row -2, next interior is row -3
    W[-1, 1:-1] = (2.0*W[-2, 1:-1] - W[-3, 1:-1]
                   - nu*(W[-2, 2:] - 2.0*W[-2, 1:-1] + W[-2, :-2]))
    # Bottom ghost (col 0)
    W[1:-1, 0] = (2.0*W[1:-1, 1] - W[1:-1, 2]
                  - nu*(W[2:, 1] - 2.0*W[1:-1, 1] + W[:-2, 1]))
    # Top ghost (col -1)
    W[1:-1, -1] = (2.0*W[1:-1, -2] - W[1:-1, -3]
                   - nu*(W[2:, -2] - 2.0*W[1:-1, -2] + W[:-2, -2]))
    # Corner condition: d²w/dxdy = 0 at each corner
    W[0,  0]  = W[2,  0]  + W[0,  2]  - W[2,  2]
    W[-1, 0]  = W[-3, 0]  + W[-1, 2]  - W[-3, 2]
    W[0,  -1] = W[2,  -1] + W[0,  -3] - W[2,  -3]
    W[-1, -1] = W[-3, -1] + W[-1, -3] - W[-3, -3]

    return W

def _laplacian_padded(W: np.ndarray, h: float) -> np.ndarray:
    """
    5-point Laplacian on the interior of a (N+3)x(N+3) padded array.
    Returns an (N+1)x(N+1) array (physical grid).
    """
    return (
        W[:-2, 1:-1] + W[2:, 1:-1]
        + W[1:-1, :-2] + W[1:-1, 2:]
        - 4.0 * W[1:-1, 1:-1]
    ) / h ** 2

def _biharmonic(w_phys: np.ndarray, h: float, nu: float) -> np.ndarray:
    """
    Apply ∇⁴ to the (N+1)x(N+1) physical grid array w_phys.
    Uses the free BC ghost-point fill, then applies ∇² twice.
    Returns an (N+1)x(N+1) array.
    """
    N = w_phys.shape[0] - 1

    # Pad physical grid into (N+3)x(N+3) array
    W = np.zeros((N + 3, N + 3))
    W[1:-1, 1:-1] = w_phys
    W = _apply_free_bc(W, nu)

    # First Laplacian: result is (N+1)x(N+1)
    lap1 = _laplacian_padded(W, h)

    # Pad lap1 and apply free BC again before second Laplacian
    L = np.zeros((N + 3, N + 3))
    L[1:-1, 1:-1] = lap1
    L = _apply_free_bc(L, nu)

    # Second Laplacian
    return _laplacian_padded(L, h)


def _laplacian_phys(w_phys: np.ndarray, h: float, nu: float) -> np.ndarray:
    """
    Laplacian on (N+1)x(N+1) physical grid with free-BC
    """
    N = w_phys.shape[0] - 1
    W = np.zeros((N + 3, N + 3))
    W[1:-1, 1:-1] = w_phys
    W = _apply_free_bc(W, nu)
    return _laplacian_padded(W, h)


def _Kh(w: np.ndarray, h: float, nu: float,
        K0: float, T: float, D: float) -> np.ndarray:
    """
    Stiffness operator: Kh w = K0*w - T*∇²w + D*∇⁴w
    """
    out = K0 * w
    if T != 0.0:
        out = out - T * _laplacian_phys(w, h, nu)
    if D != 0.0:
        out = out + D * _biharmonic(w, h, nu)
    return out


def _Bh(w: np.ndarray, h: float, nu: float,
        K1: float, T1: float) -> np.ndarray:
    """
    Damping operator: Bh w = K1*w - T1*∇²w
    """
    out = K1 * w
    if T1 != 0.0:
        out = out - T1 * _laplacian_phys(w, h, nu)
    return out


# Newmark-Beta implicit solve

def _build_system_matrix(
    N: int, h: float, nu: float,
    rho_h: float, K0: float, T: float, D: float,
    K1: float, T1: float,
    beta: float, gamma: float, dt: float,
) -> object:
    """
    Build and factorise the sparse (N+1)² x (N+1)² system matrix

        M_sys = (rho_h*I  +  beta*dt²*Kh  +  gamma*dt*Bh)

    for the NB2 acceleration solve.
    Returns a callable solve(rhs) -> solution via spsolve.
    """
    n_pts = (N + 1) ** 2

    cols, rows, vals = [], [], []
    e = np.zeros((N + 1, N + 1))

    for col_idx in range(n_pts):
        i, j = divmod(col_idx, N + 1)
        e[i, j] = 1.0

        # Apply (rho_h*I + beta*dt²*Kh + gamma*dt*Bh) to basis vector
        col_vec = (
            rho_h * e
            + beta * dt ** 2 * _Kh(e, h, nu, K0, T, D)
            + gamma * dt     * _Bh(e, h, nu, K1, T1)
        )

        nz = np.flatnonzero(col_vec)
        for row_idx in nz:
            rows.append(row_idx)
            cols.append(col_idx)
            vals.append(col_vec.flat[row_idx])

        e[i, j] = 0.0

    M = sparse.csc_matrix((vals, (rows, cols)), shape=(n_pts, n_pts))
    return M


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_kirchhoff_love(
    force_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    L: float = 0.24,
    N: int = 80,
    T: float = 1.0,
    rho_h: float = 2.7,
    E: float = 69e9,
    h_plate: float = 0.001,
    nu: float = 0.33,
    K0: float = 0.0,
    T_tension: float = 0.0,
    K1: float = 0.0,
    T1: float = 0.0,
    dt: float | None = None,
    save_path: str | None = None,
):
    """
    Simulate a 2D Kirchhoff-Love plate and return a nodal-line contour plot.

    Solves:
        rho_h * w_tt = -K0*w + T_tension*∇²w - D*∇⁴w - K1*w_t + T1*∇²w_t + F

    on [0, L] x [0, L] with free boundary conditions, using the implicit
    Newmark-Beta scheme (β=1/4, γ=1/2).

    Parameters
    ----------
    force_fn : Callable[[ndarray, ndarray, float], ndarray]
        Forcing function F(X, Y, t).  Receives 2D meshgrid arrays X, Y
        (both shape (N+1, N+1)) and scalar time t.  Returns an (N+1, N+1)
        array of force values.
        Example (small-patch centre forcing):
            def force_fn(X, Y, t):
                xc, yc = L/2, L/2
                mask = (np.abs(X - xc) <= 0.01) & (np.abs(Y - yc) <= 0.01)
                return np.where(mask, F0 * np.cos(xi * t), 0.0)

    L : float
        Side length of the square plate in metres (default 0.24 m).

    N : int
        Number of grid intervals per axis (default 80).
        Grid has (N+1)x(N+1) points.

    T : float
        Total simulation time in seconds (default 1.0 s).

    rho_h : float
        Surface mass density ρ*h in kg/m² (default 2.7 for 1 mm aluminium).

    E : float
        Young's modulus in Pa (default 69e9 for aluminium).

    h_plate : float
        Plate thickness in metres (default 0.001 m).

    nu : float
        Poisson's ratio (default 0.33 for aluminium).

    K0, T_tension, K1, T1 : float
        Generalised model coefficients (all default 0.0 for classical
        Kirchhoff-Love).

    dt : float or None
        Time step.  If None, chosen automatically as dt = 0.9 * h / 2,
        which is conservative and sufficient for the implicit scheme.

    save_path : str or None
        If provided, the figure is saved to this path.

    Returns
    -------
    The displacement array
    """
    # --- Derived parameters ---
    D = E * h_plate ** 3 / (12.0 * (1.0 - nu ** 2))   # flexural rigidity
    h = L / N                                           # grid spacing
    if dt is None:
        dt = 1 * h / 2.0                             # conservative default

    beta_nb  = 0.25   # Newmark β  (unconditionally stable with γ=0.5)
    gamma_nb = 0.50   # Newmark γ

    x = np.linspace(0.0, L, N + 1)
    y = np.linspace(0.0, L, N + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    n_pts = (N + 1) ** 2

    # --- Initial conditions (plate at rest) ---
    w = np.zeros((N + 1, N + 1))   # displacement
    v = np.zeros((N + 1, N + 1))   # velocity
    # Initial acceleration from equation of motion
    F0 = force_fn(X, Y, 0.0)
    a  = (F0 - _Kh(w, h, nu, K0, T_tension, D)
              - _Bh(v, h, nu, K1, T1)) / rho_h

    # --- Build and LU-factorise system matrix (done once) ---
    # Pre-factorising gives ~60x speedup over calling spsolve each step,
    # since the matrix is constant (linear problem, fixed dt).
    print("Building and factorising system matrix … ", end="", flush=True)
    M_sys = _build_system_matrix(
        N, h, nu, rho_h, K0, T_tension, D, K1, T1,
        beta_nb, gamma_nb, dt
    )
    lu = splu(M_sys)
    print("done.")

    # --- Time loop ---
    total_steps = max(1, int(round(T / dt)))
    t_current   = 0.0

    ts = []
    ws = []
    for step in range(total_steps):
        t_new = t_current + dt

        # Stage I: predictor (first-order extrapolation)
        w_p = w + dt * v + 0.5 * dt ** 2 * (1.0 - 2.0 * beta_nb) * a
        v_p = v + dt * (1.0 - gamma_nb) * a

        # Stage II: solve for a_new
        # (rho_h*I + beta*dt²*Kh + gamma*dt*Bh) a_new
        #     = F(t_new) - Kh*w_p - Bh*v_p
        F_new = force_fn(X, Y, t_new)
        rhs   = F_new - _Kh(w_p, h, nu, K0, T_tension, D) \
                      - _Bh(v_p, h, nu, K1, T1)

        a_new_flat = lu.solve(rhs.ravel())
        a_new = a_new_flat.reshape(N + 1, N + 1)

        # Stage III: corrector
        w = w_p + beta_nb  * dt ** 2 * a_new
        v = v_p + gamma_nb * dt      * a_new
        a = a_new

        t_current = t_new

        if (step + 1) % max(1, total_steps // 10) == 0:
            pct = 100 * (step + 1) / total_steps
            print(f"  {pct:.0f}% complete (t = {t_current:.4f} s)")
            ts.append(t_current)
            ws.append(w)
    ts.append(t_current)
    ws.append(w)
    return ts,ws
