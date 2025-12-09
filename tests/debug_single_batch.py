
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast
import logging
import sys
import os
from models.mamba_denoiser import MambaDenoiser
from models.audio_encoder import AudioEncoder
from models.embed import TextEmbedding
from models.masked_diffusion_scheduler import MaskedDiffusionScheduler
from dataset.dataloader import load_data
from utils.tokenize import load_tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_single_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    tokenizer_path = "tokenizer/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        logger.error(f"Tokenizer not found at {tokenizer_path}")
        return
    
    tokenizer, token_ids = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = token_ids["pad"]
    mask_token_id = token_ids["mask"]
    
    logger.info(f"Vocab size: {vocab_size}")

    logger.info("Loading sample data...")
    train_dl, _ = load_data(batch_size=2, tokenizer=tokenizer, sample=True, num_workers=0)

    batch = next(iter(train_dl))
    (padded_text_seqs, text_seq_lens, padded_mel_specs, mel_spec_lens, stop_token_targets) = batch
    
    input_ids = padded_text_seqs.to(device)
    mel_batch = padded_mel_specs.to(device).unsqueeze(1)
    
    logger.info(f"Input IDs shape: {input_ids.shape}") # [B, S]
    logger.info(f"Mel Batch shape: {mel_batch.shape}") # [B, 1, F, T]
    d_model = 256
    audio_encoder = AudioEncoder(in_channels=1, dropout=0.0).to(device) # No dropout for debug
    text_embedder = TextEmbedding(vocab_size=vocab_size, embed_dim=d_model, max_seq_len=512).to(device)
    model = MambaDenoiser(
        vocab_size=vocab_size,
        text_dim=d_model,
        d_model=d_model,
        audio_channels=128,
        n_layers=2,
        mask_token_id=mask_token_id,
    ).to(device)
    
    scheduler = MaskedDiffusionScheduler(num_steps=10, masking_schedule="cosine")
    
    optimizer = optim.AdamW(
        list(model.parameters()) + list(audio_encoder.parameters()) + list(text_embedder.parameters()),
        lr=1e-3
    )

    logger.info("Starting overfit loop on single batch...")
    model.train()
    audio_encoder.train()
    text_embedder.train()
    
    for step in range(100):
        optimizer.zero_grad()
        
        token_mask = (input_ids != pad_token_id).to(device)
        
        audio_features = audio_encoder(mel_batch)
        x_0, _ = text_embedder(input_ids, token_mask)
        t = torch.rand(x_0.shape[0], device=device)
        masking_rate = scheduler.get_masking_rate(t)
        corrupt_mask = torch.rand(input_ids.shape, device=device) < masking_rate.unsqueeze(1)
        # Ensure at least one token is masked if rate > 0, or handle t=0
        
        masked_input_ids = input_ids.clone()
        masked_input_ids[corrupt_mask] = mask_token_id
        
        x_masked, _ = text_embedder(masked_input_ids, token_mask)
        logits = model(
            x_t=x_masked,
            t=t,
            audio_features=audio_features,
            corrupt_mask=None
        )
        
        if corrupt_mask.sum() > 0:
            logits_flat = logits[corrupt_mask]
            targets_flat = input_ids[corrupt_mask]
            loss = F.cross_entropy(logits_flat, targets_flat)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if step % 10 == 0:
       
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1)
                
                if corrupt_mask.sum() > 0:
                    correct = (pred_ids[corrupt_mask] == input_ids[corrupt_mask]).float().sum()
                    total = corrupt_mask.sum().float()
                    acc = correct / total
                else:
                    acc = 0.0
                
                logger.info(f"Step {step}: Loss={loss.item():.4f}, GradNorm={grad_norm.item():.4f}, Masked Acc={acc:.4f}")

                if step % 20 == 0:
                    logger.info(f"Target: {tokenizer.decode(input_ids[0].tolist())}")
                    logger.info(f"Pred  : {tokenizer.decode(pred_ids[0].tolist())}")

if __name__ == "__main__":
    debug_single_batch()
