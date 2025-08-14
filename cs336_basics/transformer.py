import torch
from jaxtyping import Float, Int

from .embedding import Embedding
from .normalization import RMSNorm
from .attention import MultiHeadSelfAttention
from .rope import RotaryPositionalEmbedding
from .ffn import SwiGLU
from .linear import Linear


class TransformerBlock(torch.nn.Module):
    """A Transformer block contains two ‘sublayers’, one for the multihead self attention,
    and another for the feed-forward network.
    In each sublayer, we first perform RMSNorm, then the main operation (MHA/FF), finally adding in the
    residual connection.

    Returns tensor with the predicted UN-NORMALIZED next-word distribution for each token.
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


class TransformerLM(torch.nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = torch.nn.Sequential(*[
            TransformerBlock(
                d_model, 
                num_heads,
                d_ff,
                rope_theta,
                context_length
            ) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, in_indices: Int[torch.Tensor, " batch_size sequence_length"]):
        x = self.embedding(in_indices)
        x = self.transformer_blocks(x)
        x = self.norm(x)
        x = self.linear(x)
        return x
