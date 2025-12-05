import torch
import math


class MaskedDiffusionScheduler:
    def __init__(
        self,
        num_steps: int = 1000,
        masking_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_steps = num_steps
        self.masking_schedule = masking_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

    def get_masking_rate(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get masking rate at time t.

        Different schedules from literature:
        - Linear: Austin et al. (2021) - simple baseline
        - Cosine: Nichol & Dhariwal (2021) "Improved DDPM"
        - Sqrt: Ramesh et al. (2022) DALL-E 2


        @param t: torch.Tensor representing Timesteps [B] in range [0, 1]

        Returns:
            masking_rate: torch.Tensor of shape [B] representing probability of masking each token
        """
        if self.masking_schedule == "linear":
            return t
        elif self.masking_schedule == "cosine":
            return 1 - torch.cos(t * math.pi / 2)
        elif self.masking_schedule == "sqrt":
            return torch.sqrt(t)
        else:
            return t

    def apply_masking(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask_value: torch.Tensor = None,
    ):
        """
        Apply random masking to embeddings.

        Follows the absorbing state approach from:
        - Austin et al. "Structured Denoising Diffusion Models" (2021)
          Section 3.2: Absorbing state corruption
        - Kuleshov et al. (2022): Hybrid discrete-continuous diffusion


            @param x: Clean embeddings [B x S x E]
            @param t: Timesteps [B] in range [0, 1]
            @param mask_value: Optional mask token embedding [1 x 1 x E]

        Returns:
            x_masked: Masked embeddings [B x S x E]
            corrupt_mask: Binary mask [B x S ] where 1=corrupted
        """
        batch_size, seq_len, embed_dim = x.shape
        masking_rate = self.get_masking_rate(t)

        corrupt_mask = torch.rand(
            batch_size, seq_len, device=x.device
        ) < masking_rate.unsqueeze(1)

        if mask_value is not None:
            mask_tokens = mask_value.expand(batch_size, seq_len, -1)
            x_masked = torch.where(corrupt_mask.unsqueeze(-1), mask_tokens, x)
        else:
            x_masked = x * (~corrupt_mask).unsqueeze(-1).float()

        return x_masked, corrupt_mask
