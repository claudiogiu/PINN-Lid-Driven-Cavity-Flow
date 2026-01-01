import warnings
from typing import Tuple
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

class Network(nn.Module):
    """
    Interface for a fully-connected neural network mapping (x, y) to (psi, p)
    for use in Physics-Informed Neural Networks (PINNs) applied to the
    2D steady incompressible lid-driven cavity problem.

    Attributes:
        in_features (int): Number of input features.
        hidden_features (int): Width of each hidden layer.
        hidden_layers (int): Number of hidden layers.
        out_features (int): Number of outputs.
        x_center (float): Center used for linear normalization of x.
        x_scale (float): Scale used for linear normalization of x.
        y_center (float): Center used for linear normalization of y.
        y_scale (float): Scale used for linear normalization of y.
        net (nn.Sequential): The internal MLP with tanh activations.

    Methods:
        forward(xy: torch.Tensor) -> torch.Tensor:
            Computes (psi, p) from physical coordinates.

        from_domain(domain: Tuple[float,float,float,float], hidden_features: int, hidden_layers: int) -> "Network":
            Constructs a normalized network from a domain.

        _normalize_xy(xy: torch.Tensor) -> torch.Tensor:
            Applies linear normalization to map (x,y) into [-1,1]^2.

        _init_weights() -> None:
            Applies Xavier initialization to all linear layers.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 64,
        hidden_layers: int = 5,
        out_features: int = 2,
        x_center: float = 0.5,
        x_scale: float = 0.5,
        y_center: float = 0.5,
        y_scale: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features

        self.register_buffer("x_center", torch.tensor(float(x_center)))
        self.register_buffer("x_scale", torch.tensor(float(x_scale)))
        self.register_buffer("y_center", torch.tensor(float(y_center)))
        self.register_buffer("y_scale", torch.tensor(float(y_scale)))

        layers = [nn.Linear(in_features, hidden_features), nn.Tanh()]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    @classmethod
    def from_domain(
        cls,
        domain: Tuple[float, float, float, float],
        hidden_features: int = 64,
        hidden_layers: int = 5,
    ) -> "Network":
        x_min, x_max, y_min, y_max = domain

        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        x_scale = 0.5 * (x_max - x_min)
        y_scale = 0.5 * (y_max - y_min)

        return cls(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=2,
            x_center=x_center,
            x_scale=x_scale,
            y_center=y_center,
            y_scale=y_scale,
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        if xy.shape[-1] != 2:
            raise ValueError(f"Expected input with last dimension 2, got {xy.shape[-1]}")
        xy_norm = self._normalize_xy(xy)
        return self.net(xy_norm)

    def _normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[..., 0:1]
        y = xy[..., 1:2]
        x_norm = (x - self.x_center) / self.x_scale
        y_norm = (y - self.y_center) / self.y_scale
        return torch.cat([x_norm, y_norm], dim=-1)

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)