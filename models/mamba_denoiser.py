from models.pipeline import pipeline, load_librispeech_samples
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.masked_diffusion_scheduler import MaskedDiffusionScheduler
from models.mamba_block import MambaBlock
from utils.consts import EPS


class MambaDenoiser(nn.Module):
    """
    @param x_t: torch.Tensor representing the result of
    the  geometric noise addition of shape [B x S x E]

    @param t: diffusion timesteps of shape [B] in range [0, 1]
    @param audio_features: torch.Tensor representing encoded audio conditioning of shape [B x C x T' x F' ]
    - mask: torch.Tensor token mask of shape [B x S] where 1=valid, 0=pad
    - corrupt_mask: optional corruption mask [B x L] where 1=corrupted, 0=clean

    Returns:
    - score: torch.Tensor predicted score ∇log p(x_t|audio) of shape [B x S x E] for score matching

    Rest are hparams, will expand on them in config.py
    """

    def __init__(
        self,
        text_dim: int = 128,
        audio_channels: int = 128,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        n_layers: int = 4,
        dropout: float = 0.1,
        mask_token_id: int = -1,
        predict_scores: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.text_dim = text_dim
        self.audio_channels = audio_channels
        self.d_model = d_model
        self.n_layers = n_layers
        self.predict_scores = predict_scores
        self.mask_token_id = mask_token_id
        self.mask_token = nn.Parameter(torch.randn(1, 1, text_dim, **factory_kwargs))
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.text_proj = nn.Linear(text_dim, d_model)
        self.audio_pool = nn.AdaptiveAvgPool2d((1, None))
        self.audio_proj = nn.Linear(audio_channels, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_cross = nn.LayerNorm(d_model)
        self.mamba_blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    ngroups=ngroups,
                    layer_idx=i,
                    device=device,
                    dtype=dtype,
                )
                for i in range(n_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for l in range(n_layers)]
        )

        self.output_proj = nn.Linear(d_model, text_dim)

        if self.predict_scores:
            self.score_scale = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.SiLU(),
                nn.Linear(d_model // 4, 1),
                nn.Softplus(),
            )

    def _get_timestep_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        audio_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        corrupt_mask: torch.Tensor | None = None,
    ):

        batch_size, seq_len, _ = x_t.shape

        if corrupt_mask is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            x_input = torch.where(corrupt_mask.unsqueeze(-1).bool(), mask_tokens, x_t)
        else:
            x_input = x_t

        t_emb = self._get_timestep_embedding(t, self.d_model)
        t_emb = self.time_embed(t_emb).unsqueeze(1)

        x = self.text_proj(x_input)

        audio = self.audio_pool(audio_features).squeeze(2)
        audio = audio.transpose(1, 2)
        audio = self.audio_proj(audio)
        x = x + t_emb
        attn_out, _ = self.cross_attn(
            query=x,
            key=audio,
            value=audio,
            key_padding_mask=None,
        )
        x = self.ln_cross(x + attn_out)

        for mamba_block, ln in zip(self.mamba_blocks, self.layer_norms):
            residual = x
            x = ln(x)
            x = mamba_block(x)
            x = x + residual

        output = self.output_proj(x)
        if self.predict_scores:

            scale = self.score_scale(t_emb)
            output = output * scale

        if mask is not None:
            output = output * mask.unsqueeze(-1)

        return output

    def compute_score_loss(self, score_pred, x_t, x_0, sigma_t, mask=None):
        sigma_t = sigma_t.view(-1, 1, 1)
        # kind of a moot point to clamp, might miss something 
        sigma_t = torch.clamp(sigma_t, min=1e-5)
        true_score = -(x_t - x_0) / (sigma_t**2)
        # from song 2019
        # x = x_0 + sigma eps, eps ~ N(0, I)
        # d/dx logp_sigma(x|x_0) = -(x_t-x_0/sigma^2)

        loss = F.mse_loss(score_pred, true_score, reduction="none")

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / (mask.sum() * self.text_dim + 1e-8)
        else:
            loss = loss.mean()
        #not sure if clamping would even work tbh
        loss = torch.clamp(loss, max=1e6)

        return loss


if __name__ == "__main__":
    num_samples = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples, texts, srs = load_librispeech_samples(
        num_samples=num_samples, split="train.clean.100"
    )

    t = torch.randn(size=(num_samples,))
    print(t.size())
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = MambaDenoiser(device=device)
    model = model.to(device)
    encoded_audio, text_embeddings, diffusion_feats = pipeline(
        samples, texts, t, tokenizer, use_char_level=False, device=device
    )
    x_t, eps_true, alpha_t, sigma_t = diffusion_feats
    x_t = x_t.to(device)
    encoded_audio = encoded_audio.to(device)
    text_embeddings = text_embeddings.to(device)
    t = t.to(device)

    mask = torch.ones(num_samples, x_t.size(1), device=device)

    score_pred = model(
        x_t=x_t, t=t, audio_features=encoded_audio, mask=mask, corrupt_mask=None
    )

    loss = model.compute_score_loss(
        score_pred=score_pred, x_t=x_t, x_0=text_embeddings, sigma_t=sigma_t, mask=mask
    )

    print(f"Score prediction shape: {score_pred.shape}")
    print(f"Loss: {loss.item():.4f}")
