import torch
import sys
import os

from models.mamba_denoiser import MambaDenoiser

def test_fix():
    d_model = 256
    audio_channels = 128 # Corrected value
    model = MambaDenoiser(
        text_dim=d_model,
        d_model=d_model,
        audio_channels=audio_channels,
        n_layers=2,
    )
    
    batch_size = 2
    seq_len = 10
    # audio_features from AudioEncoder: (B, 128, Freq', Time')
    # After pool (1, None) -> (B, 128, 1, Time')
    # squeeze(2) -> (B, 128, Time')
    # transpose(1, 2) -> (B, Time', 128)
    # So input to audio_proj is (B, Time', 128)
    
    audio_features = torch.randn(batch_size, 128, 32, 16) # (B, C, F, T)
    
    # need to mock other inputs
    x_t = torch.randn(batch_size, seq_len, d_model)
    t = torch.rand(batch_size)
    
    try:
        model(x_t, t, audio_features)
        print("Forward pass successful")
    except RuntimeError as e:
        print(f"Forward pass failed: {e}")
        raise e

if __name__ == "__main__":
    test_fix()
