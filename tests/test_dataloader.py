from datasets import load_dataset
from dataset.dataloader import LibriSpeechDataset
from tokenizers import Tokenizer
import torch

hf_stream = load_dataset(
    "openslr/librispeech_asr",
    split="validation.clean", 
    streaming=True
)

hf = [x for _, x in zip(range(3), hf_stream)]

tokenizer_path = "tokenizer/tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

def tokenize_fn(text):
    return torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

ds = LibriSpeechDataset(hf, tokenization_method=tokenize_fn)

for i in range(3):
    text_seq, mel = ds[i]

    print(f"Sample {i}")
    print("Raw text:")
    print(hf[i]["text"])

    print("\nTokenized text:")
    print(text_seq)

    print("\nDecoded text:")
    print(tokenizer.decode(text_seq.tolist()))
    print("\nMel-spectrogram shape:", mel.shape)
    print("  → (mel_bins, time_steps)")
    audio = hf[i]["audio"]["array"]
    sr = hf[i]["audio"]["sampling_rate"]
    # Audio(audio, rate=sr)
