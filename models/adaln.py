import torch.nn as nn
from models.mamba_block import MambaBlock


class AdaLNZero(nn.Module):
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.scale_shift_gate = nn.Linear(cond_dim, 3 * d_model)
        nn.init.zeros_(self.scale_shift_gate.weight)
        nn.init.zeros_(self.scale_shift_gate.bias)

    def forward(self, x, cond):
        norm_x = self.norm(x)
        scale_shift_gate = self.scale_shift_gate(cond).unsqueeze(1)
        scale, shift, gate = scale_shift_gate.chunk(3, dim=-1)
        return norm_x * (1 + scale) + shift, gate


class AdaLNMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        cond_dim,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.adaln = AdaLNZero(d_model, cond_dim)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, cond):
        norm_x, gate = self.adaln(x, cond)
        mamba_out = self.mamba(norm_x)
        return x + gate * mamba_out
