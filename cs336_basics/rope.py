import torch
from jaxtyping import Float
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        is_ = torch.arange(max_seq_len).unsqueeze(1)
        ks_ = torch.arange(d_k // 2).unsqueeze(0)
        thetas = is_ / theta ** (2 * ks_ / d_k)
        cos_arr = torch.cos(thetas).to(device)
        sin_arr = torch.sin(thetas).to(device)
        self.register_buffer("cos_arr", cos_arr, persistent=False)
        self.register_buffer("sin_arr", sin_arr, persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, " ... sequence_length d_k"],
        token_positions: Float[torch.Tensor, " ... sequence_length"],
    ) -> Float[torch.Tensor, " ... sequence_length d_k"]:
        out = torch.empty_like(x)
        cos_arr = self.get_buffer("cos_arr")
        sin_arr = self.get_buffer("sin_arr")
        pos = token_positions
        out[..., ::2] = x[..., ::2] * cos_arr[pos] - x[..., 1::2] * sin_arr[pos]
        out[..., 1::2] = x[..., 1::2] * cos_arr[pos] + x[..., ::2] * sin_arr[pos]

        return out
