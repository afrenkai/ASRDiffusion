from datasets import load_dataset
from dataset.dataloader import LibriSpeechDataset
from tokenizers import Tokenizer
import torch

# Load a small HF subset (fast!)
hf_stream = load_dataset(
    "openslr/librispeech_asr",
    split="validation.clean", 
    streaming=True
)

# Convert ONLY first 3 elements into a normal list
hf = [x for _, x in zip(range(3), hf_stream)]

# Load tokenizer
tokenizer_path = "Data/tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

def tokenize_fn(text):
    return torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

ds = LibriSpeechDataset(hf, tokenization_method=tokenize_fn)

for i in range(3):
    text_seq, mel = ds[i]

    print("\n==============================")
    print(f"Sample {i}")
    print("==============================")

    # Print text
    print("Raw text:")
    print(hf[i]["text"])

    # Print tokenized sequence
    print("\nTokenized text:")
    print(text_seq)

    # Convert tokens back to text
    print("\nDecoded text:")
    print(tokenizer.decode(text_seq.tolist()))

    # Print mel details
    print("\nMel-spectrogram shape:", mel.shape)
    print("  → (mel_bins, time_steps)")

    # Listen to audio
    audio = hf[i]["audio"]["array"]
    sr = hf[i]["audio"]["sampling_rate"]
    # Audio(audio, rate=sr)
