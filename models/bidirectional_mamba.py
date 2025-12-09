import torch
import torch.nn as nn
from models.mamba_block import MambaBlock

class BidirectionalMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
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
        self.d_model = d_model
        self.forward_mamba = MambaBlock(
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
            
        self.backward_mamba = MambaBlock(
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
        
        # Projection to combine forward and backward
        # We concat them, so input to proj is 2 * d_model
        self.out_proj = nn.Linear(2 * d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, inference_params=None):

        
        out_fwd = self.forward_mamba(x) # [B, L, D]

        # Flip sequence along length dimension (dim=1), do a little <-> op
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.backward_mamba(x_rev)
        out_rev = torch.flip(out_rev, dims=[1])

        out = torch.cat([out_fwd, out_rev], dim=-1)
        out = self.out_proj(out)
        
        return out
