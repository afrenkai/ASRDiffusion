from transformers import AutoTokenizer
from models.audio_encoder import AudioEncoder
from models.forward_diffusion import Diffusion
from models.embed import TextEmbedding
from utils.consts import NUM_MELS
from dataset.conversion_utils import SpeechConverter
import torch
from utils.reproducibility import set_seed

#ds_utils swap
from datasets import load_dataset
from dataset.ds_utils import ds_use
from utils.consts import DATASET_SUBSETS

audio_encoder = AudioEncoder(in_channels=1, dropout=0.0)
speech_converter = SpeechConverter(num_mels=NUM_MELS)
text_embedding = TextEmbedding(vocab_size=50000, embed_dim=128, max_seq_len=512)


def load_librispeech_samples(num_samples=5, split="test.clean"):
    samples = []
    srs = []
    texts = []

    print(f"Loading {num_samples} samples from LibriSpeech {split} (streaming)...")
    #ds = load_dataset("openslr/librispeech_asr", split=split, streaming=True)
    ds_dict = ds_use(splir=split, subset=DATASET_SUBSETS[0], streaming=True)
    ds = [(split, DATASET_SUBSETS[0])]

    for i, item in enumerate(ds):
        if i >= num_samples:
            break

        audio_array = torch.tensor(item["audio"]["array"], dtype=torch.float32)
        sample_rate = item["audio"]["sampling_rate"]
        text = item["text"]

        samples.append(audio_array)
        texts.append(text)
        srs.append(sample_rate)
    return samples, texts, srs


def create_mels_batches(samples):
    mel_specs = []
    for i, audio in enumerate(samples):
        mel_spec = speech_converter.convert_to_mel_spec(audio)
        mel_specs.append(mel_spec)
        print(f"Sample {i+1}: Mel spectrogram shape = {mel_spec.shape} (mels × time)")

    max_time = max(mel.shape[-1] for mel in mel_specs)
    padded_mel_specs = []
    for mel in mel_specs:
        pad_amount = max_time - mel.shape[-1]
        padded_mel = torch.nn.functional.pad(
            mel, (0, pad_amount), mode="constant", value=0
        )
        padded_mel_specs.append(padded_mel.unsqueeze(0))

    mel_batch = torch.cat(padded_mel_specs, dim=0).unsqueeze(1)
    print(
        f"Padded and batched mel spectrograms: {mel_batch.shape} [Batch, Channel, Mels, Time]"
    )
    return mel_batch


def encode(encoder: AudioEncoder, mel_batch: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    encoder = encoder.to(device)
    mel_batch = mel_batch.to(device)
    encoder.eval()
    with torch.no_grad():
        encoded_audio = encoder(mel_batch)
    return encoded_audio


def char_level_tokenize(text, tokenizer):
    return tokenizer.text_to_seq_char_level(text)


def hf_tokenize(text, tokenizer, max_seq_len=512):
    encoded = tokenizer(
        text, truncation=True, max_length=max_seq_len, return_tensors="pt"
    )
    return encoded["input_ids"].squeeze(0).tolist()


def tokenize_and_embed(use_char_level: bool, texts, tokenizer, device: str = "cpu") -> torch.Tensor:
    if use_char_level:
        token_seqs = [char_level_tokenize(text, tokenizer) for text in texts]
    else:
        token_seqs = [hf_tokenize(text, tokenizer) for text in texts]
    max_len = max(len(seq) for seq in token_seqs)
    num_emb = text_embedding.tok_emb.num_embeddings
    padded_tokens = torch.zeros(len(token_seqs), max_len, dtype=torch.long)

    for i, seq in enumerate(token_seqs):
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        seq_tensor = torch.clamp(seq_tensor, max=num_emb - 1)
        padded_tokens[i, : len(seq_tensor)] = seq_tensor

    text_embedding.to(device)
    padded_tokens = padded_tokens.to(device)
    text_embedding.eval()
    with torch.no_grad():

        text_embeddings, _ = text_embedding(padded_tokens, mask=None)
    return text_embeddings


def forward_diffusion(
    timesteps: torch.Tensor,
    text_embeddings: torch.Tensor,
    mask=None,
    min_sigma: float = 0.01,
    max_sigma=50.0,
):
    timesteps = timesteps.to(text_embeddings.device)
    diffusion = Diffusion(min_sigma, max_sigma)
    return diffusion.add_noise(text_embeddings, timesteps, mask=mask)


def pipeline(samples, texts, timesteps: torch.Tensor, tokenizer, use_char_level, device: str = "cpu"):
    set_seed()
    mel_batches = create_mels_batches(samples)
    encoded_audio = encode(audio_encoder, mel_batches, device=device)
    text_embeddings = tokenize_and_embed(use_char_level, texts, tokenizer, device=device)
    diffusion_feats = forward_diffusion(timesteps, text_embeddings)
    return encoded_audio, text_embeddings, diffusion_feats
