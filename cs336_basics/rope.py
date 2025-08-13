import torch
from einops import einsum
from jaxtyping import Float


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """Construct the RoPE module and create buffers if needed

        Args:
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        
        NOTE: Try to vectorize the implementation.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_k = d_k

        rotation_matrices: list[torch.Tensor] = []
        for i in range(max_seq_len):
            sub_rotation_matrices = []
            for k in range(0, d_k // 2):
                # NOTE: why 2k instead of 2k+1 refer to assignment 1 handout
                angle = i / (theta ** ((2*k)/d_k))
                c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
                sub_rotation_matrices.append(torch.tensor([[c, -s], [s, c]]))
            rotation_matrices.append(torch.block_diag(*sub_rotation_matrices))

        self.rotation_matrices: Float[torch.Tensor, "max_seq_len d_k d_k"] = torch.stack(rotation_matrices).to(device)
        # TODO: Why register_buffer?
        self.register_buffer("rotation_matrix", self.rotation_matrices, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.
        
        Run RoPE for a given input tensor.
        Args:
            x (Float[Tensor, "... sequence_len d_k"])
            token_positions (Int[Tensor, "... sequence_len"])
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        rotation_matrices_for_tokens = self.rotation_matrices[token_positions]
        # NOTE: "seq d_k d_k, ... seq d_k -> ... seq d_k" doesn't work
        return einsum(
            rotation_matrices_for_tokens,
            x,
            "... seq d_k_1 d_k_2, ... seq d_k_2 -> ... seq d_k_1"
        )
