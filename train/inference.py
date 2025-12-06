import os
import sys
import torch
import argparse
import numpy as np
from models.mamba_denoiser import MambaDenoiser
from models.audio_encoder import AudioEncoder
from models.embed import TextEmbedding
from models.masked_diffusion_scheduler import MaskedDiffusionScheduler
from models.pipeline import load_librispeech_samples, create_mels_batches
from tokenizers import Tokenizer

#TODO: cleanup, written in a state of illness
def inference(args):
    device = args.device
    
    # Load tokenizer
    tokenizer_path = "Data/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    mask_token_id = tokenizer.token_to_id("[MASK]")
    if mask_token_id is None:
        mask_token_id = vocab_size
        vocab_size += 1
    
    audio_encoder = AudioEncoder(in_channels=1, dropout=0.1).to(device)
    text_embedder = TextEmbedding(vocab_size=vocab_size, embed_dim=args.d_model, max_seq_len=512).to(device)
    model = MambaDenoiser(
        text_dim=args.d_model,
        d_model=args.d_model,
        audio_channels=128,
        n_layers=args.n_layers,
        mask_token_id=mask_token_id
    ).to(device)
    
    scheduler = MaskedDiffusionScheduler(
        num_steps=args.num_steps,
        masking_schedule="cosine"
    )
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
        text_embedder.load_state_dict(checkpoint['text_embedder_state_dict'])
    
    model.eval()
    audio_encoder.eval()
    text_embedder.eval()

    #TODO: plug in parker's pipeline here wherever it is. 
    print("Loading test audio...")
    samples, texts, srs = load_librispeech_samples(num_samples=1, split="test.clean")
    sample = samples[0]
    ground_truth = texts[0]
    
    mel_batch = create_mels_batches([sample]).to(device)
    
    with torch.no_grad():
        audio_features = audio_encoder(mel_batch)
        #TODO: replace with MASK id from text embedder 
        # Estimate length: heuristic based on audio length (e.g., 100ms per token probably wrong idk) or some fixed interval
        seq_len = len(ground_truth.split()) + 5 # Simple heuristic that is a pos but good enough for this demo
        # tokenizer should actually have a MASK token id to use here
        batch_size = 1
        
        mask_token_emb = text_embedder.tok_emb(torch.tensor(mask_token_id, device=device)).view(1, 1, -1)
        x_t = mask_token_emb.expand(batch_size, seq_len, -1).clone()
        
        print("Starting reverse diffusion...")
        timesteps = torch.linspace(1, 0, args.num_steps, device=device)
        
        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val.item(), device=device)

            # Note: corrupt_mask is technically not needed for inference if model doesn't enforce it, 
            # but we can pass all ones if needed or None. MambaDenoiser docstring says optional.
            x_0_pred = model(x_t, t, audio_features)
            
            # Re-mask to t_next (t - 1/steps)
            if i < len(timesteps) - 1:
                t_next = timesteps[i+1]
                t_next_batch = torch.full((batch_size,), t_next.item(), device=device)
                
            #x_{t-1} ~ q(x_{t-1} | x_0_pred)
                x_t, _ = scheduler.apply_masking(x_0_pred, t_next_batch, mask_value=mask_token_emb)
            else:
                x_t = x_0_pred

        # KNN lol
        emb_weights = text_embedder.tok_emb.weight # [Vocab, E]
        
        # x_t: [1, S, E]
        # dist: [1, S, Vocab]
        # We want to find argmin ||x_t - emb||^2 = ||x_t||^2 + ||emb||^2 - 2 <x_t, emb>
        # Equivalent to argmax <x_t, emb> if embeddings are normalized
        
        logits = torch.matmul(x_t, emb_weights.t())
        predicted_ids = torch.argmax(logits, dim=-1)
        
        predicted_text = tokenizer.decode(predicted_ids[0].tolist(), skip_special_tokens=True)
        
        print(f"\nGround Truth: {ground_truth}")
        print(f"Predicted:    {predicted_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    inference(args)
