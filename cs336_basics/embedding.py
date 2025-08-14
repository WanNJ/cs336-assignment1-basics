import torch
from jaxtyping import Float, Int


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None, *args, **kwargs):
        """
        
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        """
        super().__init__(*args, **kwargs)
        self.weights = torch.nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weights, 0, 1, -3, 3)

    def forward(self, token_ids: Int[torch.Tensor, "... seq"]) -> Float[torch.Tensor, "... seq d_model"]:
        return self.weights[token_ids]
