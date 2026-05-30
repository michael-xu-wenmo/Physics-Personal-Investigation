"""
kirchhoff_love_animated.py
--------------------------
Extends the Kirchhoff-Love plate solver with an animated version that shows
how the displacement field and nodal lines evolve through time.

The animation runs the full simulation first, collecting frames at regular
intervals, then renders them — this is much faster than re-computing inside
FuncAnimation's draw callback.

Each frame shows two side-by-side panels:
  Left  — filled contour of displacement w(x, y, t)
  Right — zero contour (nodal lines) of w(x, y, t)

Output formats: .mp4 (ffmpeg) or .gif (pillow).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from scipy import sparse
from scipy.sparse.linalg import splu
from typing import Callable

# Re-use all spatial operators from the existing solver
from test import (
    _Kh,
    _Bh,
    _build_system_matrix,
)


def animate_kirchhoff_love(
    force_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    L: float = 0.24,
    N: int = 80,
    T: float = 10.0,
    rho_h: float = 2.7,
    E: float = 69e9,
    h_plate: float = 0.001,
    nu: float = 0.33,
    K0: float = 0.0,
    T_tension: float = 0.0,
    K1: float = 0.0,
    T1: float = 0.0,
    dt: float | None = None,
    n_frames: int = 120,
    fps: int = 30,
    save_path: str = "kirchhoff_love_animation.mp4",
) -> animation.FuncAnimation:
    """
    Simulate a Kirchhoff-Love plate and save an animation of the evolving
    displacement field and nodal lines.

    The full simulation is run first with frames saved at regular intervals,
    then the animation is assembled and written to disk.

    Parameters
    ----------
    force_fn : Callable[[ndarray, ndarray, float], ndarray]
        Forcing function F(X, Y, t).  Receives meshgrid arrays X, Y
        (shape (N+1, N+1)) and scalar time t.  Returns (N+1, N+1) array.

    L : float
        Plate side length in metres (default 0.24 m).

    N : int
        Number of grid intervals per axis (default 80).

    T : float
        Total simulation time in seconds (default 1.0 s).

    rho_h : float
        Surface mass density ρ·h in kg/m² (default 2.7).

    E : float
        Young's modulus in Pa (default 69e9 for aluminium).

    h_plate : float
        Plate thickness in metres (default 0.001 m).

    nu : float
        Poisson's ratio (default 0.33).

    K0, T_tension, K1, T1 : float
        Generalised model coefficients (default 0.0).

    dt : float or None
        Time step in seconds.  Defaults to 0.9 * h / 2.

    n_frames : int
        Number of animation frames to capture (default 120).
        More frames → smoother animation but larger file.

    fps : int
        Frames per second in the output video (default 30).

    save_path : str
        Output file path.  Use ".mp4" for video or ".gif" for GIF.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The assembled animation object (also saved to save_path).
    """
    # ------------------------------------------------------------------ setup
    D = E * h_plate**3 / (12.0 * (1.0 - nu**2))
    h = L / N
    if dt is None:
        dt = 0.9 * h / 2.0

    beta_nb = 0.25
    gamma_nb = 0.50

    x = np.linspace(0.0, L, N + 1)
    y = np.linspace(0.0, L, N + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # --------------------------------------------------------- initial state
    w = np.zeros((N + 1, N + 1))
    v = np.zeros((N + 1, N + 1))
    F_init = force_fn(X, Y, 0.0)
    a = (F_init - _Kh(w, h, nu, K0, T_tension, D) - _Bh(v, h, nu, K1, T1)) / rho_h

    # ------------------------------------------- build & factorise (once)
    print("Building and factorising system matrix … ", end="", flush=True)
    M_sys = _build_system_matrix(
        N,
        h,
        nu,
        rho_h,
        K0,
        T_tension,
        D,
        K1,
        T1,
        beta_nb,
        gamma_nb,
        dt,
    )
    lu = splu(M_sys)
    print("done.")

    # -------------------------------------------------- run & collect frames
    total_steps = max(1, int(round(T / dt)))
    # Capture a frame every `stride` steps so we get exactly n_frames
    stride = max(1, total_steps // n_frames)
    actual_frames = total_steps // stride

    frames_w = []  # displacement snapshots
    frames_t = []  # corresponding times
    t_current = 0.0

    print(
        f"Running {total_steps} steps, capturing every {stride} "
        f"→ {actual_frames} frames …"
    )

    for step in range(total_steps):
        t_new = t_current + dt

        # Newmark-Beta stage I: predictor
        w_p = w + dt * v + 0.5 * dt**2 * (1.0 - 2.0 * beta_nb) * a
        v_p = v + dt * (1.0 - gamma_nb) * a

        # Stage II: solve for acceleration
        F_new = force_fn(X, Y, t_new)
        rhs = F_new - _Kh(w_p, h, nu, K0, T_tension, D) - _Bh(v_p, h, nu, K1, T1)
        a_new = lu.solve(rhs.ravel()).reshape(N + 1, N + 1)

        # Stage III: corrector
        w = w_p + beta_nb * dt**2 * a_new
        v = v_p + gamma_nb * dt * a_new
        a = a_new
        t_current = t_new

        # Capture frame
        if step % stride == 0:
            frames_w.append(w.copy())
            frames_t.append(t_current)

        if (step + 1) % max(1, total_steps // 10) == 0:
            pct = 100 * (step + 1) / total_steps
            print(
                f"  {pct:.0f}%  (t = {t_current:.4f} s, "
                f"frames captured: {len(frames_w)})"
            )

    print(f"Simulation complete. Assembling {len(frames_w)}-frame animation …")

    # ------------------------------------------------------ build animation
    # Determine a symmetric colour limit from the max displacement seen
    # (use the 99th percentile to avoid outlier spikes dominating the scale)
    all_max = np.percentile(
        [np.abs(fw).max() for fw in frames_w if np.abs(fw).max() > 0], 99
    )
    if all_max == 0:
        all_max = 1.0
    vlim = all_max

    fig, (ax_disp, ax_node) = plt.subplots(
        1,
        2,
        figsize=(11, 5),
        gridspec_kw={"wspace": 0.35},
    )

    # --- Left panel: displacement field ---
    cf = ax_disp.contourf(
        x,
        y,
        frames_w[0],
        levels=50,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
    )
    cbar = fig.colorbar(cf, ax=ax_disp, fraction=0.046, pad=0.04)
    cbar.set_label("w  (m)", fontsize=9)
    ax_disp.set_title("Displacement  w(x, y, t)", fontsize=10)
    ax_disp.set_xlabel("x (m)")
    ax_disp.set_ylabel("y (m)")
    ax_disp.set_aspect("equal")

    # --- Right panel: nodal lines ---
    try:
        cn = ax_node.contour(
            x, y, frames_w[0], levels=[0], colors="steelblue", linewidths=1.5
        )
    except Exception:
        cn = None
    ax_node.set_xlim(0, L)
    ax_node.set_ylim(0, L)
    ax_node.set_title("Nodal lines  (w = 0)", fontsize=10)
    ax_node.set_xlabel("x (m)")
    ax_node.set_ylabel("y (m)")
    ax_node.set_aspect("equal")

    time_text = fig.suptitle(f"t = {frames_t[0]:.4f} s", fontsize=11)

    def _update(frame_idx: int):
        """Update both panels for frame frame_idx."""
        w_frame = frames_w[frame_idx]
        t_frame = frames_t[frame_idx]

        # --- Redraw displacement contourf ---
        ax_disp.cla()
        ax_disp.contourf(
            x,
            y,
            w_frame,
            levels=50,
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
        )
        ax_disp.set_title("Displacement  w(x, y, t)", fontsize=10)
        ax_disp.set_xlabel("x (m)")
        ax_disp.set_ylabel("y (m)")
        ax_disp.set_aspect("equal")

        # --- Redraw nodal lines ---
        ax_node.cla()
        ax_node.set_xlim(0, L)
        ax_node.set_ylim(0, L)
        if np.abs(w_frame).max() > 0:
            try:
                ax_node.contour(
                    x, y, w_frame, levels=[0], colors="steelblue", linewidths=1.5
                )
            except Exception:
                pass
        ax_node.set_title("Nodal lines  (w = 0)", fontsize=10)
        ax_node.set_xlabel("x (m)")
        ax_node.set_ylabel("y (m)")
        ax_node.set_aspect("equal")

        time_text.set_text(f"t = {t_frame:.4f} s")
        return []

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frames_w),
        interval=1000 // fps,
        blit=False,
    )

    # ----------------------------------------------------------- save
    ext = save_path.rsplit(".", 1)[-1].lower()
    if ext == "gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="h264",
            extra_args=["-pix_fmt", "yuv420p"],  # broad compatibility
        )

    print(f"Writing to {save_path} …", end=" ", flush=True)
    anim.save(save_path, writer=writer, dpi=120)
    print("done.")
    plt.close(fig)

    return anim


# ---------------------------------------------------------------------------
# Example — same aluminium plate as kirchhoff_love.py, animated over 1 s
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    L = 0.24

    f1 = 995.0
    xi = 2.0 * np.pi * f1
    F0 = 1e10
    xc, yc = L / 2.0, L / 2.0

    def force_fn(X, Y, t):
        mask = (np.abs(X - xc) <= 0.01) & (np.abs(Y - yc) <= 0.01)
        return np.where(mask, F0 * np.cos(xi * t), 0.0)

    animate_kirchhoff_love(
        force_fn=force_fn,
        save_path="kirchhoff_love_animation.mp4",
    )
