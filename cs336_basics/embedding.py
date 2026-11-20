import torch
import torch.nn as nn
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeding: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embs = nn.Parameter(torch.empty((num_embeding, embedding_dim), device=device, dtype=dtype))

    def forward(self, token_ids: Int[torch.Tensor, "bs seq_len"]) -> Float[torch.Tensor, "bs seq_len d_emb"]:
        token_ids = token_ids.to(self.embs)
        return self.embs[token_ids]
