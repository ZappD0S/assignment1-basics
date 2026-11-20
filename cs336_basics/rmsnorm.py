import torch
from jaxtyping import Float
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: Float[torch.Tensor, "... d"]) -> Float[torch.Tensor, "... d"]:
        in_dtype = x.dtype

        x = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(x**2, -1, keepdim=True) / self.d_model + self.eps)
        rmsnorm = x / rms * self.g

        rmsnorm = rmsnorm.to(in_dtype)
        return rmsnorm
