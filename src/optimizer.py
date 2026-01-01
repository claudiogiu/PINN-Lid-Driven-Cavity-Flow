import time
import warnings
from typing import Any, Callable, Dict, Optional
import torch
from torch import nn
from torch.optim import Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import _LRScheduler

warnings.filterwarnings("ignore")

class Optimizer:
    """
    Interface for training Physics-Informed Neural Networks (PINNs) using Adam with mini-batching. 

    Attributes:
        pinn (Any): The PINN object providing the model and loss computation.
        model (nn.Module): The neural network to be optimized.
        device (torch.device): Device on which the model is placed.
        optimizer (torch.optim.Optimizer): Adam optimizer instance.
        scheduler (_LRScheduler, optional): Optional learning-rate scheduler.
        num_steps (int): Total number of Adam iterations.
        lr (float): Learning rate.
        weight_decay (float): L2 regularization coefficient.
        n_interior (int): Number of PDE points per batch.
        n_boundary (int): Number of boundary points per batch.
        log_every (int): Logging frequency.
        eval_every (int): Evaluation frequency.

    Methods:
        train(eval_hook=None, verbose=True) -> None:
            Executes the training loop and monitors training progress.

        _single_step(step: int) -> Dict[str, float]:
            Performs one Adam update and returns loss metrics.
    """

    def __init__(
        self,
        pinn: Any,
        num_steps: int = 30000,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        n_interior: int = 4096,
        n_boundary: int = 512,
        log_every: int = 500,
        eval_every: int = 0,
        optimizer: Optional[TorchOptimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
    ) -> None:
        self.pinn = pinn
        self.num_steps = num_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.log_every = log_every
        self.eval_every = eval_every

        if not hasattr(pinn, "model"):
            raise AttributeError("The provided PINN must expose a 'model' attribute.")

        if not isinstance(pinn.model, nn.Module):
            raise TypeError("pinn.model must be an instance of nn.Module.")

        self.model = pinn.model

        if hasattr(pinn, "device"):
            self.device = torch.device(pinn.device)
        else:
            self.device = next(self.model.parameters()).device

        self.model.to(self.device)

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

    def _single_step(self, step: int) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)

        loss, components = self.pinn.compute_loss(
            n_interior=self.n_interior,
            n_boundary=self.n_boundary,
        )

        if not torch.is_tensor(loss):
            raise TypeError("compute_loss must return a scalar torch.Tensor as first value.")

        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        metrics: Dict[str, float] = {"loss": float(loss.detach().cpu().item())}

        if isinstance(components, dict):
            for key, val in components.items():
                if torch.is_tensor(val):
                    metrics[key] = float(val.detach().cpu().item())
                else:
                    try:
                        metrics[key] = float(val)
                    except Exception:
                        pass

        return metrics

    def train(
        self,
        eval_hook: Optional[Callable[[int, nn.Module, Any], None]] = None,
        verbose: bool = True,
    ) -> None:
        start_time = time.time()
        last_log_time = start_time

        if verbose:
            print(
                f"Starting PINN training for {self.num_steps} steps...\n"
                f"Learning rate:{self.lr}\n"
                f"Number of Interior Points: {self.n_interior}\n"
                f"Number of Boundary Points: {self.n_boundary}\n"
            )

        for step in range(1, self.num_steps + 1):
            metrics = self._single_step(step)

            if verbose and (step == 1 or step % self.log_every == 0 or step == self.num_steps):
                now = time.time()
                elapsed = now - start_time
                dt = now - last_log_time
                last_log_time = now

                msg = f"[{step:05d}] loss={metrics['loss']:.6e} "
                for key in sorted(metrics.keys()):
                    if key != "loss":
                        msg += f"{key}={metrics[key]:.6e} "
                msg += f"(elapsed={elapsed:.1f}s, step={dt:.1f}s)"
                print(msg)

            if eval_hook is not None and self.eval_every > 0:
                if step % self.eval_every == 0 or step == self.num_steps:
                    eval_hook(step, self.model, self.pinn)

        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {total_time:.1f} seconds.")