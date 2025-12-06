import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from models.mamba_denoiser import MambaDenoiser
from models.audio_encoder import AudioEncoder
from models.embed import TextEmbedding
from models.masked_diffusion_scheduler import MaskedDiffusionScheduler
from models.pipeline import (
    load_librispeech_samples,
    create_mels_batches,
    tokenize_and_embed,
)
from tokenizers import Tokenizer

# Added imports by AT
from dataset.dataloader import LibriSpeechDataset, get_data_loader
from datasets import load_dataset
from utils.consts import PAD


# TODO: handle device mismatches preemptively here, do NOT want to be walkint through call stacks later


def train(args):
    device = args.device
    tokenizer_path = "Data/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id(PAD)
    
    # Reserve last token for masking if not present, or use a specific one
    mask_token_id = tokenizer.token_to_id("[MASK]")
    if mask_token_id is None:
        print("Warning: [MASK] token not found in tokenizer. Adding a virtual mask token at the end of vocabulary.")
        mask_token_id = vocab_size
        vocab_size += 1 # Increase vocab size to accommodate the mask token

    audio_encoder = AudioEncoder(in_channels=1, dropout=0.1).to(device)
    text_embedder = TextEmbedding(
        vocab_size=vocab_size, embed_dim=args.d_model, max_seq_len=512
    ).to(device)

    model = MambaDenoiser(
        text_dim=args.d_model,
        d_model=args.d_model,
        audio_channels=128,
        n_layers=args.n_layers,
        mask_token_id=mask_token_id
    ).to(device)

    scheduler = MaskedDiffusionScheduler(
        num_steps=args.num_steps, masking_schedule="cosine"
    )

    optimizer = optim.AdamW(
        list(model.parameters())
        + list(audio_encoder.parameters())
        + list(text_embedder.parameters()),
        lr=args.lr,
    )

    # Data Loading
    # TODO: ds, dl. rn this is a hacky method since idk what is being done w.r.t datasets and dataloaders, integration is left to someone else.
    # samples, texts, srs = load_librispeech_samples(num_samples=args.batch_size * 10, split="train.clean.100")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    # AT Imp
    if args.sample:
        print(
            "[INFO] Sample mode active: using hf-internal-testing/librispeech_asr_demo"
        )
        hf_train = load_dataset(
            "hf-internal-testing/librispeech_asr_demo", split="validation"
        )
        args.batch_size = min(2, args.batch_size)
        args.epochs = 1
        args.device = "cuda"
        args.num_steps = 10
    else:
        print("Loading data...")
        hf_train = load_dataset("openslr/librispeech_asr", "clean", "train.100")

    def tokenize_fn(text):
        return torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

    train_ds = LibriSpeechDataset(hf_train, tokenization_method=tokenize_fn)
    train_dl = get_data_loader(train_ds, args.batch_size, shuffle=True, num_workers=2, pad_token_id=pad_token_id)

    # ENTER THE TRAINING ARENA!!!!
    model.train()
    audio_encoder.train()
    text_embedder.train()

    print(f"Starting training for {args.epochs} epochs...")

    # ============================================================
    # NOTE TO artem:
    #
    # speech_collate_fn returns:
    #   padded_text_seqs, text_seq_lens,
    #   padded_mel_specs, mel_spec_lens,
    #   stop_token_targets
    #
    # Currently, I ONLY USE:
    #   padded_text_seqs (character IDs)
    #   padded_mel_specs (mel spectrograms)
    #
    # I IGNORE:
    #   text_seq_lens
    #   mel_spec_lens
    #   stop_token_targets
    #
    # This is intentional to keep training minimal.
    # You can integrate these later for more advanced
    # masking, variable-length supervised loss, or decoder training.
    # ============================================================

    for epoch in range(args.epochs):
      
        # num_batches = len(samples) // args.batch_size

        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            """
            batch_samples = samples[i*args.batch_size : (i+1)*args.batch_size]
            batch_texts = texts[i*args.batch_size : (i+1)*args.batch_size]
            mel_batch = create_mels_batches(batch_samples).to(device)
            audio_features = audio_encoder(mel_batch) # [B, C, T, F]
            # Note: tokenize_and_embed needs to be compatible with our manual embedding if we want to use the mask token embedding
           
            encoded = tokenizer(batch_texts, padding=True, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encoded.input_ids.to(device)
            """

            (
                padded_text_seqs,
                text_seq_lens,
                padded_mel_specs,
                mel_spec_lens,
                stop_token_targets,
            ) = batch

            input_ids = padded_text_seqs.to(device)
            token_mask = (input_ids != pad_token_id).to(device)

            mel_batch = padded_mel_specs.to(device).unsqueeze(1)

            audio_features = audio_encoder(mel_batch)

            x_0, _ = text_embedder(input_ids, token_mask)

            mask_token_emb = text_embedder.tok_emb(
                torch.tensor(mask_token_id, device=device)
            ).view(1, 1, -1)

            t = torch.rand(x_0.shape[0], device=device)
            x_masked, corrupt_mask = scheduler.apply_masking(
                x_0, t, mask_value=mask_token_emb
            )

            """
            x_0 = text_embedder.tok_emb(input_ids)
            x_0 = x_0 + text_embedder.pos_enc[:x_0.size(1), :].unsqueeze(0)
            mask_token_emb = text_embedder.tok_emb(torch.tensor(mask_token_id, device=device)).view(1, 1, -1)
            t = torch.rand(x_0.shape[0], device=device)
            x_masked, corrupt_mask = scheduler.apply_masking(x_0, t, mask_value=mask_token_emb)
            """

            predicted_embeddings = model(
                x_t=x_masked,
                t=t,
                audio_features=audio_features,
                corrupt_mask=corrupt_mask,
            )

            loss = torch.nn.functional.mse_loss(
                predicted_embeddings[corrupt_mask], x_0[corrupt_mask]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "audio_encoder_state_dict": audio_encoder.state_dict(),
                    "text_embedder_state_dict": text_embedder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoint_epoch_{epoch+1}.pt",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run a small sample for quick GPU test using 73-entry demo dataset",
    )

    args = parser.parse_args()
    train(args)
