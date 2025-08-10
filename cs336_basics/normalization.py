import torch
from einops import reduce, einsum

class RMSNorm(torch.nn.Module):
    """Performs Root Mean Square normalization."""
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weights = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_dtype = x.dtype
        rms = reduce(
            torch.square(x.to(torch.float32)), 
            "batch_size sequence_length d_model -> batch_size sequence_length 1", 
            "sum"
        )
        rms = torch.sqrt(rms * (1 / self.d_model) + self.eps)
        result = einsum(self.weights, x / rms, "d_model, ... d_model -> ... d_model")
        return result.to(in_dtype)
