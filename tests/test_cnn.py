import pytest
import torch

from models.audio_encoder import AudioEncoder


def testPipeline():
    fake_data = torch.randn(4, 1, 16000, 80)
    model = AudioEncoder()
    y = model(fake_data)
    print(y.size())
    assert y.size() == torch.Size([4, 128, 2000, 10])
