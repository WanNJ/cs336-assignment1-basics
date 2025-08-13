import torch
from jaxtyping import Float, Bool
from einops import einsum, rearrange

from .linear import Linear
from .rope import RotaryPositionalEmbedding

def softmax(x: torch.Tensor, dim: int):
    """Your function should
    take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input
    tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will
    now have a normalized probability distribution."""
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    exp_x_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / exp_x_sum


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
):
    """Notes on mask: Each row i of this boolean matrix indicates which keys the query
    i should attend to. Canonically (and slightly confusingly), a value of True at position (i, j) indicates that
    the query i does attend to the key j, and a value of False indicates that the query does not attend to the
    key. In other words, “information flows” at (i, j) pairs with value True.
    """
    d_k = Q.shape[-1]
    Qt_K = einsum(Q, K, "... seq1 d_k, ... seq2 d_k -> ... seq1 seq2")
    Qt_K = Qt_K / torch.sqrt(torch.tensor([d_k]))
    # NOTE: ... selects the extra dimensions, works even if mask only has two dimensions.
    Qt_K[..., ~mask] = Qt_K[..., ~mask] - torch.inf
    return einsum(
        softmax(Qt_K, len(Qt_K.shape) - 1),
        V,
        "... seq1 seq2, ... seq2 d_v -> ... seq1 d_v"
    )


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None):
        """Module to perform causal multi-head self-attention.

            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.W_Q = Linear(d_model,  d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)

    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
        """
        seq_len = in_features.shape[-2]
        W_QKV = torch.cat([self.W_Q.weights, self.W_K.weights, self.W_V.weights])
        QKV = einsum(W_QKV, in_features, "d_qkv d_in, ... seq d_in -> ... seq d_qkv")

        # Chunk by the last dimension.
        Q, K, V = QKV.chunk(3, -1)

        # Add multi-head dimensions to Q, K, V.
        Q = rearrange(
            Q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        K = rearrange(
            K, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        V = rearrange(
            V, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )

        if self.rope is not None:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        casual_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        output = scaled_dot_product_attention(Q, K, V, casual_mask)
        output = rearrange(
            output, "... h seq_len d_head ->  ... seq_len (h d_head)"
        )
        return self.W_O.forward(output)
