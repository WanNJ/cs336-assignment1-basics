import torch
from jaxtyping import Float


from .normalization import RMSNorm
from .attention import MultiHeadSelfAttention
from .rope import RotaryPositionalEmbedding
from .ffn import SwiGLU


class TransformerBlock(torch.nn.Module):
    """A Transformer block contains two ‘sublayers’, one for the multihead self attention,
    and another for the feed-forward network.
    In each sublayer, we first perform RMSNorm, then the main operation (MHA/FF), finally adding in the
    residual connection.
    """
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len):
        super().__init__()
        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)
        self.multihead_attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
            RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        )
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[torch.Tensor, "batch sequence_length d_model"]) -> Float[torch.Tensor, " batch sequence_length d_model"]:
        y = x + self.multihead_attention(self.rms_norm1(x))
        return y + self.ffn(self.rms_norm2(y))
