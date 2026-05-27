import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy import sparse
import scipy.sparse.linalg as spla
from display import Display

from typing import Literal


def eigenwaves1D(N: int, k=10):
    # Define dx
    x = np.linspace(0, 1, N)
    dx = np.diff(x)[0]

    # Build Laplacian
    main_diag = 2 * np.ones_like(x)
    side_diag = -1 * np.ones(len(main_diag) - 1)

    evalue, efunc = eigh_tridiagonal(main_diag, side_diag)
    return np.sqrt(evalue)[:k] / dx, efunc.T[:k]


def eigenwaves2D(N: int, k=10, bound: Literal["Dirichlet", "Neumann"] = "Dirichlet"):
    # Define A
    x = np.linspace(0, 1, N)
    dx = np.diff(x)[0]

    diag = np.ones(N)
    diags = np.array([-diag, 2 * diag, -diag])
    if bound == "Neumann":
        diags[2][1] = -2
        diags[0][-2] = -2
    D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)  # one dimensional H matrix
    T = sparse.kronsum(D, D)  # two dimensional H matrix
    eigenvalues, eigenvectors = sparse.linalg.eigsh(T, k=k, which="SM")
    waves = [eigenvector.reshape((N, N)) for eigenvector in eigenvectors.T]

    return np.sqrt(eigenvalues) / dx, waves


def build_free_bc_biharmonic(N: int, nu: float = 0.3):
    """
    Biharmonic operator with free-plate BCs via ghost nodes.
    nu: Poisson's ratio (0.3 is typical for steel/aluminium)
    """
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]

    # --- 1D second-derivative FD matrix (interior only, free BCs via ghost nodes) ---
    e = np.ones(N)
    # Standard tridiagonal for d^2/dx^2
    D2 = sparse.spdiags([e, -2 * e, e], [-1, 0, 1], N, N).toarray()

    # Free BC: ghost node gives d^2w/dn^2 = 0 at boundary
    # => w[-1] = w[1], w[N] = w[N-2]  (reflecting ghost nodes)
    D2[0, 0] = -2
    D2[0, 1] = 2  # forward ghost: w_{-1} = w_{1}
    D2[-1, -1] = -2
    D2[-1, -2] = 2  # backward ghost: w_{N} = w_{N-2}
    D2 = sparse.csr_matrix(D2) / dx**2

    # --- 2D biharmonic via Kronecker: ∇⁴ = (I⊗D2 + D2⊗I)² ---
    I = sparse.eye(N)
    L = sparse.kron(I, D2) + sparse.kron(D2, I)  # ∇²
    B = L @ L  # ∇⁴

    return B, dx


def eigenwaves2D_plate(N: int, k: int = 10, nu: float = 0.3):
    B, dx = build_free_bc_biharmonic(N, nu)

    # Shift-invert mode: much faster for smallest eigenvalues
    # Small sigma avoids the zero rigid-body modes (shift away from 0)
    eigenvalues, eigenvectors = spla.eigsh(B, k=k + 6, which="LM", sigma=1e-6)

    # Discard near-zero rigid-body modes (translations/rotation of free plate)
    mask = eigenvalues > 1e-4
    eigenvalues = eigenvalues[mask][:k]
    eigenvectors = eigenvectors[:, mask][:, :k]

    # Frequencies: ω ∝ λ^(1/2) for the plate equation (λ already from ∇⁴)
    frequencies = np.sqrt(np.abs(eigenvalues))
    waves = [eigenvectors[:, i].reshape((N, N)) for i in range(k)]
    return frequencies, waves


if __name__ == "__main__":
    N = 100
    k = 40
    v, f = eigenwaves2D(N, k, bound="Neumann")
    # v, f = eigenwaves2D_plate(N, k)
    d = Display(v, f, "Chladni")
    d.export(d.plot_energy(), "freq.png")
    i = 0
    for fig in d.plot_wavefunction():
        d.export(fig, f"pattern{i}.png")
        plt.close()
        i += 1
