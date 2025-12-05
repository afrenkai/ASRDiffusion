from datasets import load_dataset
from dataset.dataloader import LibriSpeechDataset
from dataset.char_tokenization import CharacterTokenization
from IPython.display import Audio

# Load a small HF subset (fast!)
hf_stream = load_dataset(
    "openslr/librispeech_asr",
    split="validation.clean", 
    streaming=True
)

# Convert ONLY first 3 elements into a normal list
hf = [x for _, x in zip(range(3), hf_stream)]

ds = LibriSpeechDataset(hf)

tok = CharacterTokenization()

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
    print(tok.seq_to_text(text_seq))

    # Print mel details
    print("\nMel-spectrogram shape:", mel.shape)
    print("  → (mel_bins, time_steps)")

    # Listen to audio
    audio = hf[i]["audio"]["array"]
    sr = hf[i]["audio"]["sampling_rate"]
    Audio(audio, rate=sr)
