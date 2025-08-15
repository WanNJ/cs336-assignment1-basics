from collections.abc import Callable, Iterable
import math

import torch
import numpy as np
from jaxtyping import Float, Int

from einops import rearrange


def cross_entropy(inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]):
    # Subtract max value to make it numerical stable.
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    log_softmax = inputs.exp().sum(-1, keepdim=True).log() - inputs

    # Gather predicted log probabilities by index.
    targets = rearrange(targets, "batch -> batch ()")
    target_log_probs = log_softmax.gather(-1, targets)

    target_log_probs = rearrange(target_log_probs, "batch 1 -> batch")
    return target_log_probs.mean()


def get_lr_cosine_schedule(t: int, alpha_max, alpha_min, Tw, Tc):
    """In training Transformers, it is typical to use a learning rate schedule, where we start with a bigger learning
    rate, making quicker updates in the beginning, and slowly decay it to a smaller value as the model trains
    """
    if t < Tw:
        return t * alpha_max / Tw
    elif Tw <= t <= Tc:
        return alpha_min + (1 + np.cos(np.pi * (t - Tw) / (Tc - Tw))) * (alpha_max - alpha_min) / 2
    else:
        return alpha_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    p_grads = [p.grad for p in parameters if p.grad is not None]
    l2_norm = torch.sqrt(sum([torch.sum(g.pow(2)) for g in p_grads]))

    if l2_norm > max_l2_norm:
        scale_down_factor = max_l2_norm / (l2_norm + 1e-6)
        for g in p_grads:
            g *= scale_down_factor


class SGD(torch.optim.Optimizer):
    """Implements Stochastic Gradient Descent."""
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        # A dict containing default values of optimization options (used when a parameter group doesn’t specify them).
        defaults = {
            "alpha": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "gamma": weight_decay,
            "epsilon": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # Get the hyper parameters of the optimizer.
            alpha = group["alpha"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            gamma = group["gamma"]
            epsilon = group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get/Initialize state variables.
                # self.state is a default dict that returns an empty dictionary.
                state = self.state[p]
                t = state.get("t", 1) 
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                g: torch.Tensor = p.grad.data
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g.pow(2)
                alpha_t = alpha * np.sqrt(1 - np.power(beta2, t)) / (1 - np.power(beta1, t))

                # Update weight tensor in-place.
                p.data -= alpha_t * m / torch.sqrt(v + epsilon)
                # Apply weight decay.
                p.data -= alpha * gamma * p

                # Update state variables.
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


# Experiment.
if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=10)
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
