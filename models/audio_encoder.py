from typing import Callable
import torch
import torch.nn as nn


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
    ):
        super().__init__()
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
        """
        @param x: tensor of shape [B x C x T x F]

        Returns:
            tensor of shape [B x C x T' x F']

        """

        x = self.dropout_1(self.enc_act_fn_1(self.enc_norm_1(self.conv1(x))))
        x = self.dropout_2(self.enc_act_fn_2(self.enc_norm_2(self.conv2(x))))
        x = self.enc_act_fn_3(self.enc_norm_3(self.conv3(x)))
        return x
