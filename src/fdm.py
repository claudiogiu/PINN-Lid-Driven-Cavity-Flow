import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Dict, List

warnings.filterwarnings("ignore")

@dataclass
class LidDrivenCavityFDM:
    """
    Interface for solving the 2D steady incompressible lid-driven cavity flow
    using the Finite Difference Method (FDM) in the vorticity-streamfunction
    formulation.

    Attributes:
        Nx (int): Number of grid points along each spatial dimension.
        L (float): Physical domain size.
        wall_velocity (float): Lid velocity applied at the top boundary.
        rho (float): Fluid density.
        mu (float): Dynamic viscosity corresponding to Re = 100.
        dt (float): Time step for pseudo-time iteration.
        max_iter (int): Maximum number of iterations for steady-state convergence.
        tol (float): Convergence tolerance based on the L-infinity norm of vorticity updates.

    Methods:
        __post_init__() -> None:
            Initializes grid spacing, coordinate arrays, and index slices.

        solve() -> Dict[str, np.ndarray]:
            Executes the FDM solver and returns all numerical fields.

        _apply_boundary_conditions(omega: np.ndarray, psi: np.ndarray) -> None:
            Applies vorticity boundary conditions for the lid-driven cavity.

        _update_vorticity(omega: np.ndarray, omega_prev: np.ndarray,
                          psi: np.ndarray) -> None:
            Performs one iteration of the vorticity transport equation.

        _update_streamfunction(psi: np.ndarray, omega: np.ndarray) -> None:
            Solves the Poisson equation for the streamfunction.

        _compute_velocity(psi: np.ndarray) -> (np.ndarray, np.ndarray):
            Computes velocity components from the streamfunction.
    """

    Nx: int = 64
    L: float = 1.0
    wall_velocity: float = 1.0
    rho: float = 1.0
    mu: float = 0.01  
    dt: float = 1e-3
    max_iter: int = 150000
    tol: float = 1e-7

    h: float = field(init=False)
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    im: slice = field(init=False)
    i: slice = field(init=False)
    ip: slice = field(init=False)
    jm: slice = field(init=False)
    j: slice = field(init=False)
    jp: slice = field(init=False)

    def __post_init__(self) -> None:
        self.h = self.L / (self.Nx - 1)
        self.x = np.linspace(0, self.L, self.Nx)
        self.y = np.linspace(0, self.L, self.Nx)

        self.im = slice(0, self.Nx - 2)
        self.i = slice(1, self.Nx - 1)
        self.ip = slice(2, self.Nx)
        self.jm = slice(0, self.Nx - 2)
        self.j = slice(1, self.Nx - 1)
        self.jp = slice(2, self.Nx)

    def solve(self) -> Dict[str, np.ndarray]:
        omega = np.zeros((self.Nx, self.Nx))
        psi = np.zeros_like(omega)
        omega_prev = np.zeros_like(omega)
        errors: List[float] = []

        for it in range(self.max_iter):

            self._apply_boundary_conditions(omega, psi)

            omega_prev[:, :] = omega
            self._update_vorticity(omega, omega_prev, psi)
            self._update_streamfunction(psi, omega)

            if it > 10:
                err = np.linalg.norm(omega - omega_prev, ord=np.inf)
                errors.append(err)
                if err < self.tol:
                    break

        u, v = self._compute_velocity(psi)

        return {
            "u": u,
            "v": v,
            "psi": psi,
            "omega": omega,
            "errors": np.array(errors),
            "x": self.x,
            "y": self.y,
        }

    def _apply_boundary_conditions(self, omega: np.ndarray, psi: np.ndarray) -> None:
        omega[:, -1] = -2 * psi[:, -2] / self.h**2 - 2 * self.wall_velocity / self.h
        omega[:,  0] = -2 * psi[:,  1] / self.h**2
        omega[0,  :] = -2 * psi[1,  :] / self.h**2
        omega[-1, :] = -2 * psi[-2, :] / self.h**2

    def _update_vorticity(self, omega: np.ndarray, omega_prev: np.ndarray,
                          psi: np.ndarray) -> None:

        omega[self.i, self.j] = (
            omega_prev[self.i, self.j] +
            (
                -(psi[self.i, self.jp] - psi[self.i, self.jm]) / (2*self.h)
                * (omega_prev[self.ip, self.j] - omega_prev[self.im, self.j]) / (2*self.h)
                +
                (psi[self.ip, self.j] - psi[self.im, self.j]) / (2*self.h)
                * (omega_prev[self.i, self.jp] - omega_prev[self.i, self.jm]) / (2*self.h)
                +
                self.mu/self.rho * (
                    omega_prev[self.ip, self.j] +
                    omega_prev[self.im, self.j] +
                    omega_prev[self.i, self.jp] +
                    omega_prev[self.i, self.jm] -
                    4 * omega_prev[self.i, self.j]
                ) / self.h**2
            ) * self.dt
        )

    def _update_streamfunction(self, psi: np.ndarray, omega: np.ndarray) -> None:
        psi[self.i, self.j] = (
            omega[self.i, self.j] * self.h**2 +
            psi[self.ip, self.j] +
            psi[self.im, self.j] +
            psi[self.i, self.jp] +
            psi[self.i, self.jm]
        ) / 4

    def _compute_velocity(self, psi: np.ndarray) -> (np.ndarray, np.ndarray):
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)

        u[1:-1, -1] = self.wall_velocity

        u[self.i, self.j] = (psi[self.i, self.jp] - psi[self.i, self.jm]) / (2 * self.h)
        v[self.i, self.j] = -(psi[self.ip, self.j] - psi[self.im, self.j]) / (2 * self.h)

        return u, v
    

if __name__ == "__main__":
    solver: LidDrivenCavityFDM = LidDrivenCavityFDM()
    results: Dict[str, np.ndarray] = solver.solve()