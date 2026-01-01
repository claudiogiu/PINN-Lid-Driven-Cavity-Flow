import torch
import warnings
import torch.nn as nn
from typing import Tuple
import random
import numpy as np
from network import Network
from pinn import LidDrivenCavityPINN
from optimizer import Optimizer

warnings.filterwarnings("ignore")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)

    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: nn.Module = Network.from_domain(
        domain=domain,
        hidden_features=64,
        hidden_layers=5
    )

    pinn: LidDrivenCavityPINN = LidDrivenCavityPINN(
        model=model,
        domain=domain,
        Re=100.0,
        U0=1.0,
        lambda_pde=1.0,
        lambda_bc=10.0,
        device=device,
        dtype=torch.float32
    )

    opt: Optimizer = Optimizer(
        pinn=pinn,
        num_steps=30000,
        lr=1e-3,
        n_interior=4096,
        n_boundary=512,
        log_every=500,
        eval_every=0
    )

    opt.train(verbose=True)

    pinn.save_model("pinn_lid_driven_cavity.pth")