import torch
from mamba_ssm import Mamba2
from transformers import AutoTokenizer
from models.audio_encoder import AudioEncoder
from models.forward_diffusion import Diffusion
from models.embed import TextEmbedding
from utils.consts import NUM_MELS
from dataset.conversion_utils import SpeechConverter
from models.pipeline import pipeline, load_librispeech_samples
from transformers import AutoTokenizer

# from utils.tokenize import quick_tokenizer, tokenize

# https://github.com/state-spaces/mamba


class MambaDenoiser(Mamba2):
    """
    I think this should take in

    - x_t
    - eps
    - audio tensor
    - alpha_t
    - sigma_t
    - mask

    we should sample t from T ... 0

    find alpha_t and sigma_t using the formulas we have

    add noise using the function

    pass through the denoiser

    compare


    """

    def __init__(
        self, dim: int, d_state: int, d_conv: int, expand_factor: int, device: str
    ):
        self.d_model = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand_factor
        self.device = device
        # https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py
        # self.conv_init = conv_init

    def forward(
        self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None
    ):
        pass


if __name__ == "__main__":

    samples, texts, srs = load_librispeech_samples(
        num_samples=5, split="train.clean.100"
    )
    t = torch.tensor([0.1, 0.2, 0.3, 0.6, 0.8])
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    x = pipeline(samples, texts, t, tokenizer, use_char_level=False)

    # print(x)
    encoded_audio, text_embeddings, diffusion_feats = (x[0], x[1], x[2])
    # print(f"alpha_t = 4, {x[2][2][4]:.4f}")
