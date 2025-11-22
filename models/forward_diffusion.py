import torch
from utils.consts import EPS
import math


class Diffusion:
    def __init__(self, min_sigma: float = 0.1, max_sigma: float = 50.0):
        self.sigma_min = min_sigma
        self.sigma_max = max_sigma

    def noise_schedule(self, t: torch.Tensor):
        """
        t should be of shape [B] and should be in [0,1].

        Returns:
        alpha_t of shape [B] which is a signa coeff
        and sigma_t also of shape [B] which is a noise coeff
        """

        # https://arxiv.org/pdf/2102.09672, eqn 17
        s = EPS
        f_t = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
        f_0 = math.cos(s / (1 + s) * (math.pi / 2)) ** 2

        alpha_t_squared = f_t / f_0
        alpha_t = torch.sqrt(alpha_t_squared)
        sigma_t = torch.sqrt(1 - alpha_t_squared)
        return alpha_t, sigma_t

    def add_noise(self, x_0, t, mask=None):
        """
        add noises to uncorrupted embeddings (definitely a better word for that ) according to diffusion scheduler
        similar to song, et. al this implements x_t = alpha * x_0 + sigma_t * epsilon

        @param x_0: torch.Tensor of shape [B x S x E], which is the result of us passing it through an embedding.
        @param t: torch.Tensor of shape [B], which is the diffusion time in [0,1]
        @param mask: optional torch.Tensor (1 = valid, 0 = pad, presumably since I haven't actually written it yet)

        Returns:

            x_t: torch.Tensor of shape[B x S x E], which is a noised embedding of text
            eps: torch.Tensor of shape [B x S x E, which a tensor of the noises that were added (reusing the variable from ddpm, consider it a collection of eps_t if you will.
            alpha_t: torch.Tensor of shape [B] which has all of the signal coeffs, same as in DDPM
            sigma_t: torch.Tensor of shape [B] which has all of the noise coeffs, same as in DDPM

        """
        if x_0.ndim == 1:
            bs = x_0.shape
        else:
            bs, seq_len, emb_dim = x_0.shape
        print(bs)
        alpha_t, sigma_t = self.noise_schedule(t)
        print(alpha_t.shape)
        print(sigma_t.shape)
        alpha_t = alpha_t.view(bs, 1, 1)
        sigma_t = sigma_t.view(bs, 1, 1)

        eps = torch.randn_like(x_0)

        if mask is not None:
            eps = eps * mask.unsqueeze(-1)

        # using formula from song https://arxiv.org/pdf/2205.14217
        x_t = alpha_t * x_0 + sigma_t * eps
        return x_t, eps, alpha_t.squeeze(), sigma_t.squeeze()

    def denoise(self, x_t, eps, alpha_t, sigma_t):
        """
        denoises using like algebra lol
        also taken from song paper. idt this is good enough, but will serve as a basis for what I'll do later.
        eps = (x_t - alpha * x_0) / sigma_t
        x_0 = (x_t - sigma_t * eps) / alpha_t

        @param x_t: torch.Tensor of shape [B x S x E] which are the noised text embeddings
        @param eps: torch.tensor of shape [B x S x E], which is the predicted noise from prior step
        @param alpha_t: torch.Tensor of shape [B], signal coeffs, same as in DDPM
        @param sigma_t, torch.Tensor of shape [B], noise coeffs, same as in DDPM

        Returns

            x_0_hat: torch.Tensor of shape [B x S x E], which are predicted denoised embeddings
        """

        alpha_t = alpha_t.view(-1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1)
        x_0_hat = (x_t - sigma_t * eps) / alpha_t
        return x_0_hat
