# pip install scikit-fem
from skfem import MeshTri, Basis, BilinearForm
import numpy as np
from scipy.sparse.linalg import eigsh
from skfem.element import ElementTriArgyris
from display import Display
import matplotlib.pyplot as plt


def eigenwaves2D_fem(
    n_elements: int = 30, k: int = 10, nu: float = 0.3, grid_n: int = 200
):
    mesh = MeshTri.init_tensor(
        np.linspace(0, 1, n_elements + 1), np.linspace(0, 1, n_elements + 1)
    )

    e = ElementTriArgyris()
    basis = Basis(mesh, e)

    @BilinearForm
    def biharmonic(u, v, _):
        return (
            u.hess[0, 0] * v.hess[0, 0]
            + nu * (u.hess[0, 0] * v.hess[1, 1] + u.hess[1, 1] * v.hess[0, 0])
            + 2 * (1 - nu) * u.hess[0, 1] * v.hess[0, 1]
            + u.hess[1, 1] * v.hess[1, 1]
        )

    @BilinearForm
    def inertia(u, v, _):
        return u * v

    K = biharmonic.assemble(basis)
    M = inertia.assemble(basis)

    eigenvalues, eigenvectors = eigsh(K, k=k + 6, M=M, which="LM", sigma=1e-6)

    mask = eigenvalues > 1e-6
    eigenvalues = eigenvalues[mask][:k]
    eigenvectors = eigenvectors[:, mask][:, :k]

    # Interpolate each eigenvector onto a regular grid for plotting
    grid_x = np.linspace(0, 1, grid_n)
    grid_y = np.linspace(0, 1, grid_n)
    xx, yy = np.meshgrid(grid_x, grid_y)
    points = np.array([xx.ravel(), yy.ravel()])
    P = basis.probes(points)  # maps DOF values -> grid point values

    waves = [
        (P @ eigenvectors[:, i]).reshape(grid_n, grid_n)
        for i in range(eigenvalues.shape[0])
    ]

    return np.sqrt(np.abs(eigenvalues)), waves


if __name__ == "__main__":
    N = 100
    k = 40
    v, f = eigenwaves2D_fem(N, k)
    d = Display(v, f, "Chladni")
    d.export(d.plot_energy(), "freq.png")
    i = 0
    for fig in d.plot_wavefunction():
        d.export(fig, f"pattern{i}.png")
        plt.close()
        i += 1
