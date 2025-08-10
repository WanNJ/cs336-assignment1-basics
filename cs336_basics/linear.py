import numpy as np
import torch
from einops import einsum


class Linear(torch.nn.Module):
    """Linear transformation."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        """This function should accept the following parameters:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))
        sigma = np.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weights, 0, sigma, -3 * sigma, 3 * sigma)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # => x @ self.weights.T
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")
