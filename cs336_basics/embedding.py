import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = torch.nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weights, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
