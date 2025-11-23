from typing import Callable
import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """
    Augmentations for the spectrogram.

    Masks out in a triu along both the frequency and time dimension, respectively
    Literally might make 0 difference, will fid out in ablation

    @param freq_mask_param: int  how often to apply the random mask in the spectrogram along the frequency dimension
    @param time_mask_param: int how often to apply the random mask along the time dim (should be axis = 4 (3 if we are 0 indexed))

    """

    def __init__(
        self, freq_mask_param=27, time_mask_param=70, n_freq_masks=2, n_time_masks=2
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        if not self.training:
            return x

        batch_size, channels, freq, time = x.shape

        for mask in range(self.n_freq_masks):
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            f0 = torch.randint(0, freq - f, (1,)).item()
            x[:, :, f0 : f0 + f, :] = 0

        for mask in range(self.n_time_masks):
            t = torch.randint(0, self.time_mask_param, (1,)).item()
            t0 = torch.randint(0, time - t, (1,)).item()
            x[:, :, :, t0 : t0 + t] = 0

        return x


class AudioEncoder(nn.Module):
    """
    Assuming the pipeline gives me an audio tensor of shape [Batch x Channels x Time x Frames]
    The reason it's set to 1 is because most mel specs only have 1 channel. everything intermediate is fair game I think
    This is a CNN Encoder. I stole the nums from resnet. These are malleable except the input channels obvs.
    Outputs Tensor of Batch x Channels x Time' x Frames'
    """

    def __init__(
        self,
        in_channels: int = 1,
        act_fn: Callable = nn.GELU,
        norm: Callable = nn.BatchNorm2d,
        dropout: float = 0.0,
        use_spec_augment: bool = True,
    ):
        super().__init__()
        self.spec_augment = SpecAugment() if use_spec_augment else None
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2)
        self.enc_norm_1 = norm(32)
        self.enc_act_fn_1 = act_fn()
        self.dropout_1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.enc_norm_2 = norm(64)
        self.enc_act_fn_2 = act_fn()
        self.dropout_2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.enc_norm_3 = norm(128)
        self.enc_act_fn_3 = act_fn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.spec_augment is not None:
            x = self.spec_augment(x)

        x = self.dropout_1(self.enc_act_fn_1(self.enc_norm_1(self.conv1(x))))
        x = self.dropout_2(self.enc_act_fn_2(self.enc_norm_2(self.conv2(x))))
        x = self.enc_act_fn_3(self.enc_norm_3(self.conv3(x)))
        return x
