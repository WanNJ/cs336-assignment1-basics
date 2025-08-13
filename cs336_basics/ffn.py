import torch
from jaxtyping import Float

from .linear import Linear


class SwiGLU(torch.nn.Module):
    """Implements a SwiGLU feed-forward network."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        w1_x = self.w1(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        w3_x = self.w3(x)
        return self.w2(silu_w1_x * w3_x)
