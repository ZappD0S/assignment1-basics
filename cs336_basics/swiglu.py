import torch
from jaxtyping import Float
from torch import nn

from .linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff, device=None, dtype=None):
        super().__init__()

        # TODO: in the doc it says "You should set 𝑑ff to approximately 8/3 * 𝑑model in your implementation"
        # should both be input parameters then?

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[torch.Tensor, " ... d_model"]):
        def _silu(x):
            return x * torch.sigmoid(x)

        return self.w2(_silu(self.w1(x)) * self.w3(x))
