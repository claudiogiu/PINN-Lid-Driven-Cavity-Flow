import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

class LidDrivenCavityPINN:
    """
    Interface for a Physics-Informed Neural Network (PINN) solving the
    2D steady incompressible lid-driven cavity flow using the streamfunction-
    pressure formulation.

    Attributes:
        model (nn.Module): Neural network implementing the mapping (x, y) to (psi, p).
        domain (Tuple[float,float,float,float]): Physical domain (xmin,xmax,ymin,ymax).
        Re (float): Reynolds number.
        U0 (float): Lid velocity at the top boundary.
        lambda_pde (float): Weight for PDE residuals in the total loss.
        lambda_bc (float): Weight for boundary residuals in the total loss.
        device (torch.device): Device used for all tensors.
        dtype (torch.dtype): Floating-point precision.
        nu (torch.Tensor): Kinematic viscosity.
        x_min, x_max, y_min, y_max (torch.Tensor): Domain bounds as tensors.

    Methods:
        compute_loss(n_interior, n_boundary):
            Computes the total loss together with the PDE and boundary-condition components.

        save_model(path: str):
            Saves the neural network weights to the specified path.

        _prepare_domain_tensors():
            Converts domain parameters into device tensors.

        _grad(outputs, inputs):
            Computes gradients batch-wise.

        _compute_uvp(xy):
            Computes u, v, p from the network outputs.

        _pde_residuals(xy_int):
            Computes Navier-Stokes PDE residuals.

        _bc_residuals(xy_bc, u_target, v_target):
            Computes boundary-condition residuals.

        sample_interior(n_points):
            Generates interior collocation points within the physical domain.

        sample_boundary(n_points):
            Generates boundary collocation points on the four domain edges.
    """

    def __init__(
        self,
        model: nn.Module,
        domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        Re: float = 100.0,
        U0: float = 1.0,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.domain = domain
        self.Re = Re
        self.U0 = U0
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.device = torch.device(device)
        self.dtype = dtype

        self.model.to(self.device, dtype=self.dtype)
        self._prepare_domain_tensors()

    def _prepare_domain_tensors(self) -> None:
        x_min, x_max, y_min, y_max = self.domain
        self.x_min = torch.tensor(x_min, device=self.device, dtype=self.dtype)
        self.x_max = torch.tensor(x_max, device=self.device, dtype=self.dtype)
        self.y_min = torch.tensor(y_min, device=self.device, dtype=self.dtype)
        self.y_max = torch.tensor(y_max, device=self.device, dtype=self.dtype)

        L = x_max - x_min
        nu_val = self.U0 * L / self.Re
        self.nu = torch.tensor(nu_val, device=self.device, dtype=self.dtype)
        self.U0 = torch.tensor(self.U0, device=self.device, dtype=self.dtype)

    def sample_interior(self, n_points: int) -> torch.Tensor:
        r = torch.rand(n_points, 2, device=self.device, dtype=self.dtype)
        x = self.x_min + (self.x_max - self.x_min) * r[:, 0:1]
        y = self.y_min + (self.y_max - self.y_min) * r[:, 1:2]
        xy = torch.cat([x, y], dim=-1)
        xy.requires_grad_(True)
        return xy

    def sample_boundary(self, n_points: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_side = max(1, n_points // 4)

        t_left = torch.rand(n_side, 1, device=self.device, dtype=self.dtype)
        t_right = torch.rand(n_side, 1, device=self.device, dtype=self.dtype)
        t_bottom = torch.rand(n_side, 1, device=self.device, dtype=self.dtype)
        t_top = torch.rand(n_side, 1, device=self.device, dtype=self.dtype)

        x_left = self.x_min.expand_as(t_left)
        y_left = self.y_min + (self.y_max - self.y_min) * t_left

        x_right = self.x_max.expand_as(t_right)
        y_right = self.y_min + (self.y_max - self.y_min) * t_right

        y_bottom = self.y_min.expand_as(t_bottom)
        x_bottom = self.x_min + (self.x_max - self.x_min) * t_bottom

        y_top = self.y_max.expand_as(t_top)
        x_top = self.x_min + (self.x_max - self.x_min) * t_top

        x_bc = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
        y_bc = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)

        xy_bc = torch.cat([x_bc, y_bc], dim=-1)
        xy_bc.requires_grad_(True)

        u_left = torch.zeros(n_side, 1, device=self.device, dtype=self.dtype)
        v_left = torch.zeros_like(u_left)

        u_right = torch.zeros(n_side, 1, device=self.device, dtype=self.dtype)
        v_right = torch.zeros_like(u_right)

        u_bottom = torch.zeros(n_side, 1, device=self.device, dtype=self.dtype)
        v_bottom = torch.zeros_like(u_bottom)

        u_top = self.U0 * torch.ones(n_side, 1, device=self.device, dtype=self.dtype)
        v_top = torch.zeros_like(u_top)

        u_target = torch.cat([u_left, u_right, u_bottom, u_top], dim=0)
        v_target = torch.cat([v_left, v_right, v_bottom, v_top], dim=0)

        return xy_bc, u_target, v_target

    @staticmethod
    def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(-1)
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grads

    def _compute_uvp(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        psi_p = self.model(xy)
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]

        psi_grad = self._grad(psi, xy)
        psi_x = psi_grad[:, 0:1]
        psi_y = psi_grad[:, 1:2]

        u = psi_y
        v = -psi_x

        return u, v, p

    def _pde_residuals(self, xy_int: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u, v, p = self._compute_uvp(xy_int)

        u_grad = self._grad(u, xy_int)
        v_grad = self._grad(v, xy_int)
        p_grad = self._grad(p, xy_int)

        u_x = u_grad[:, 0:1]
        u_y = u_grad[:, 1:2]
        v_x = v_grad[:, 0:1]
        v_y = v_grad[:, 1:2]
        p_x = p_grad[:, 0:1]
        p_y = p_grad[:, 1:2]

        u_xx_yy = self._grad(u_x, xy_int)[:, 0:1] + self._grad(u_y, xy_int)[:, 1:2]
        v_xx_yy = self._grad(v_x, xy_int)[:, 0:1] + self._grad(v_y, xy_int)[:, 1:2]

        conv_u = u * u_x + v * u_y
        conv_v = u * v_x + v * v_y

        r1 = conv_u + p_x - self.nu * u_xx_yy
        r2 = conv_v + p_y - self.nu * v_xx_yy

        return r1, r2

    def _bc_residuals(
        self,
        xy_bc: torch.Tensor,
        u_target: torch.Tensor,
        v_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u, v, _ = self._compute_uvp(xy_bc)
        ru = u - u_target
        rv = v - v_target
        return ru, rv

    def compute_loss(
        self,
        n_interior: int,
        n_boundary: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mse = F.mse_loss

        xy_int = self.sample_interior(n_interior)
        r1, r2 = self._pde_residuals(xy_int)
        loss_pde = mse(r1, torch.zeros_like(r1)) + mse(r2, torch.zeros_like(r2))

        xy_bc, u_t, v_t = self.sample_boundary(n_boundary)
        ru, rv = self._bc_residuals(xy_bc, u_t, v_t)
        loss_bc = mse(ru, torch.zeros_like(ru)) + mse(rv, torch.zeros_like(rv))

        loss = self.lambda_pde * loss_pde + self.lambda_bc * loss_bc

        components: Dict[str, Any] = {
            "pde": loss_pde,
            "bc": loss_bc,
        }

        return loss, components

    def save_model(self, filename: str = "pinn_lid_driven_cavity.pth") -> None:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")