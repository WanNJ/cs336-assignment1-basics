import torch
from jaxtyping import Float
from einops import einsum


class SwiGLU(torch.nn.Module):
    """Implements a SwiGLU feed-forward network."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1_weight = torch.nn.Parameter(torch.zeros(d_ff, d_model))
        self.w2_weight = torch.nn.Parameter(torch.zeros(d_model, d_ff))
        self.w3_weight = torch.nn.Parameter(torch.zeros(d_ff, d_model))

    def forward(self, x: Float[torch.Tensor, " ... d_model"]) -> Float[torch.Tensor, " ... d_model"]:
        w1_x = einsum(self.w1_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        w3_x = einsum(self.w3_weight, x, "d_ff d_model, ... d_model -> ... d_ff")
        return einsum(self.w2_weight, silu_w1_x * w3_x, "d_model d_ff, ... d_ff -> ... d_model")
