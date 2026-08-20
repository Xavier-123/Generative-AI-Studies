"""Educational, explicit AdamW update and mixed-precision gradient helpers."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch


class ManualAdamW(torch.optim.Optimizer):
    """AdamW with every parameter-update operation visible in Python.

    It deliberately implements only the features Calibra uses: dense gradients,
    two betas, epsilon, and decoupled weight decay.  FP16/BF16 parameters keep an
    FP32 master copy so small updates are not lost before casting back.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """Apply one hand-written AdamW update to every parameter with a gradient."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("ManualAdamW does not support sparse gradients")

                state = self.state[parameter]
                state_dtype = (
                    torch.float32
                    if parameter.dtype in {torch.float16, torch.bfloat16}
                    else parameter.dtype
                )
                if not state:
                    state["step"] = 0
                    state["master_parameter"] = parameter.detach().to(state_dtype).clone()
                    state["exp_avg"] = torch.zeros_like(state["master_parameter"])
                    state["exp_avg_sq"] = torch.zeros_like(state["master_parameter"])

                state["step"] += 1
                step = state["step"]
                master_parameter = state["master_parameter"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                gradient = parameter.grad.detach().to(state_dtype)

                # Decoupled weight decay: theta <- (1 - lr * wd) * theta
                if weight_decay:
                    master_parameter.mul_(1.0 - lr * weight_decay)

                # First and second raw moments.
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                # Bias-correct both moments before applying the adaptive update.
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                master_parameter.addcdiv_(exp_avg, denominator, value=-step_size)

                parameter.copy_(master_parameter.to(dtype=parameter.dtype))

        return loss


class ManualLossScaler:
    """Small dynamic loss scaler used by the explicit FP16 backward path."""

    def __init__(
        self,
        enabled: bool,
        initial_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.enabled = enabled
        self.scale = initial_scale if enabled else 1.0
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._finite_steps = 0

    @torch.no_grad()
    def unscale_and_check_(self, parameters: Iterable[Any]) -> bool:
        """Unscale accumulated gradients and report whether NaN/Inf occurred."""
        found_nonfinite = False
        inverse_scale = 1.0 / self.scale
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if self.enabled:
                gradient.mul_(inverse_scale)
            if not bool(torch.isfinite(gradient).all().item()):
                found_nonfinite = True
        return found_nonfinite

    def update(self, found_nonfinite: bool) -> None:
        if not self.enabled:
            return
        if found_nonfinite:
            self.scale = max(1.0, self.scale * self.backoff_factor)
            self._finite_steps = 0
            return
        self._finite_steps += 1
        if self._finite_steps >= self.growth_interval:
            self.scale *= self.growth_factor
            self._finite_steps = 0


@torch.no_grad()
def manual_clip_grad_norm_(parameters: Iterable[Any], max_norm: float, eps: float = 1e-6):
    """Clip the global L2 gradient norm without ``torch.nn.utils``."""
    parameters = list(parameters)
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0)

    device = gradients[0].device
    squared_norm = torch.zeros((), device=device, dtype=torch.float32)
    for gradient in gradients:
        squared_norm.add_(gradient.detach().float().pow(2).sum())
    total_norm = squared_norm.sqrt()
    clip_coefficient = max_norm / (total_norm + eps)
    if bool((clip_coefficient < 1.0).item()):
        for gradient in gradients:
            gradient.mul_(clip_coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm


def create_manual_optimizer(model: Any, config) -> ManualAdamW:
    """Build the same decay/no-decay groups as the regular optimizer factory."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    return ManualAdamW(
        [
            {"params": decay, "weight_decay": config.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.training.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
