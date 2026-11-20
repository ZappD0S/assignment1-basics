import torch
from einops import einsum
from jaxtyping import Float
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        sigma = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, 0, sigma, -3 * sigma, 3 * sigma)

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        y: torch.Tensor = einsum(x, self.weight, "... in_feat, out_feat in_feat -> ... out_feat")

        return y
