import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import grad_scaler, autocast
from tqdm import tqdm
import argparse
import logging
import torch.nn.functional as F
from models.mamba_denoiser import MambaDenoiser
from models.audio_encoder import AudioEncoder
from models.embed import TextEmbedding
from models.masked_diffusion_scheduler import MaskedDiffusionScheduler
from tokenizers import Tokenizer
from dataset.dataloader import load_data
from utils.consts import PAD
from utils.metrics import WERMetric
from utils.distributed import setup_distributed, cleanup_distributed
from utils.tokenize import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train(args):
    rank, local_rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        logger.info(f"Training on {world_size} GPUs" if is_distributed else "Training on single device")

    tokenizer_path = args.tokenizer_path if hasattr(args, 'tokenizer_path') else "tokenizer/tokenizer.json"
    tokenizer, token_ids = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    pad_token_id = token_ids["pad"]
    mask_token_id = token_ids["mask"]
    eos_token_id = token_ids["eos"]

    audio_encoder = AudioEncoder(in_channels=1, dropout=0.1).to(device)
    text_embedder = TextEmbedding(
        vocab_size=vocab_size, embed_dim=args.d_model, max_seq_len=512
    ).to(device)

    model = MambaDenoiser(
        vocab_size=vocab_size,
        text_dim=args.d_model,
        d_model=args.d_model,
        audio_channels=128,
        n_layers=args.n_layers,
        mask_token_id=mask_token_id,
    ).to(device)

    if is_distributed:
        audio_encoder = DDP(audio_encoder, device_ids=[local_rank])
        text_embedder = DDP(text_embedder, device_ids=[local_rank])
        model = DDP(model, device_ids=[local_rank])

    scheduler = MaskedDiffusionScheduler(
        num_steps=args.num_steps, masking_schedule="cosine"
    )

    optimizer = optim.AdamW(
        list(model.parameters())
        + list(audio_encoder.parameters())
        + list(text_embedder.parameters()),
        lr=args.lr,
    )
    scaler = grad_scaler.GradScaler()

    if args.sample:
        if rank == 0:
            if args.verbose:
                print("[INFO] demo mode active: using hf-internal-testing/librispeech_asr_demo")
        args.batch_size = min(2, args.batch_size)
        args.epochs = 500
        args.num_steps = 10
    elif rank == 0:
        print("Loading data...")

    train_dl, val_dl = load_data(
        args.batch_size, 
        tokenizer, 
        sample=args.sample, 
        num_workers=4, 
        distributed=is_distributed
    )

    model.train()
    audio_encoder.train()
    text_embedder.train()

    if rank == 0:
        print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        if is_distributed:
            train_dl.sampler.set_epoch(epoch)
            
        if rank == 0:
            pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
            iterator = pbar
        else:
            iterator = train_dl

        for batch in iterator:
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

            optimizer.zero_grad()

            with autocast('cuda', enabled=True):
                audio_features = audio_encoder(mel_batch)
                
                embedder_module = text_embedder.module if is_distributed else text_embedder
                
                x_0, _ = embedder_module(input_ids, token_mask)

                t = torch.rand(x_0.shape[0], device=device)

                masking_rate = scheduler.get_masking_rate(t)
                corrupt_mask = torch.rand(input_ids.shape, device=device) < masking_rate.unsqueeze(1)
                
                masked_input_ids = input_ids.clone()
                masked_input_ids[corrupt_mask] = mask_token_id
                x_masked, _ = embedder_module(masked_input_ids, token_mask)

                predicted_logits = model(
                    x_t=x_masked,
                    t=t,
                    audio_features=audio_features,
                    corrupt_mask=None, # Disable internal masking to preserve PE
                )

                # Calculate Cross Entropy Loss on masked tokens only
                # predicted_logits: [B, S, V]
                # input_ids: [B, S]
                # corrupt_mask: [B, S]
                logits_flat = predicted_logits[corrupt_mask]
                targets_flat = input_ids[corrupt_mask]
                
                loss = F.cross_entropy(logits_flat, targets_flat)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                pbar.set_postfix({"loss": loss.item(), "grad": grad_norm.item()})
                if pbar.n % 100 == 0 and args.verbose:
                    logger.info(f"Step {pbar.n}: Loss={loss.item():.4f}, GradNorm={grad_norm.item():.4f}")

        validate(model, audio_encoder, text_embedder, scheduler, val_dl, tokenizer, device, args, mask_token_id, rank, is_distributed, eos_token_id)
        if args.save_freq:
            if rank == 0 and (epoch + 1) % args.save_freq == 0:
                model_state = model.module.state_dict() if is_distributed else model.state_dict()
                audio_state = audio_encoder.module.state_dict() if is_distributed else audio_encoder.state_dict()
                embedder_state = text_embedder.module.state_dict() if is_distributed else text_embedder.state_dict()
                
                torch.save(
                    {
                        "model_state_dict": model_state,
                        "audio_encoder_state_dict": audio_state,
                        "text_embedder_state_dict": embedder_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    f"checkpoint_epoch_{epoch+1}.pt",
                )
                
    cleanup_distributed()


def validate(model, audio_encoder, text_embedder, scheduler, val_dl, tokenizer, device, args, mask_token_id, rank, is_distributed, eos_token_id):
    model.eval()
    audio_encoder.eval()
    text_embedder.eval()
    
    metric = WERMetric()
    
    if rank == 0:
        print("Running validation...")
        
    embedder_module = text_embedder.module if is_distributed else text_embedder
    
    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            if i >= 2: 
                break
                
            (
                padded_text_seqs,
                text_seq_lens,
                padded_mel_specs,
                mel_spec_lens,
                stop_token_targets,
            ) = batch
            
            input_ids = padded_text_seqs.to(device)
            mel_batch = padded_mel_specs.to(device).unsqueeze(1)
            
            with autocast('cuda'):
                audio_features = audio_encoder(mel_batch)
                
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                
                mask_token_emb = embedder_module.tok_emb(torch.tensor(mask_token_id, device=device)).view(1, 1, -1)
                x_t = mask_token_emb.expand(batch_size, seq_len, -1).clone()
                
                timesteps = torch.linspace(1, 0, args.num_steps, device=device)
                
                for t_idx, t_val in enumerate(timesteps):
                    t = torch.full((batch_size,), t_val.item(), device=device)
                    logits = model(x_t, t, audio_features)
                    
                    pred_ids = torch.argmax(logits, dim=-1)
                    x_0_pred_emb, _ = embedder_module(pred_ids, None)
                    
                    if t_idx < len(timesteps) - 1:
                        t_next = timesteps[t_idx+1]
                        t_next_batch = torch.full((batch_size,), t_next.item(), device=device)
                        x_t, _ = scheduler.apply_masking(x_0_pred_emb, t_next_batch, mask_value=mask_token_emb)
                    else:
                        x_t = x_0_pred_emb
                        final_logits = logits
                
                predicted_ids = torch.argmax(final_logits, dim=-1)
            
            predictions = []
            references = []
            
            for j in range(batch_size):
                pred_seq = predicted_ids[j].tolist()
                ref_seq = input_ids[j].tolist()
                
                if eos_token_id in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(eos_token_id)]
                
                pred_text = tokenizer.decode(pred_seq, skip_special_tokens=True)
                ref_text = tokenizer.decode(ref_seq, skip_special_tokens=True)
                
                predictions.append(pred_text)
                references.append(ref_text)
                
            metric.update(predictions, references)
    wer = metric.total_wer / metric.count if metric.count > 0 else 0.0
    cer = metric.total_cer / metric.count if metric.count > 0 else 0.0
    
    if rank == 0:
        print(f"Validation WER: {wer:.2f}")
        print(f"Validation CER: {cer:.2f}")
    
    model.train()
    audio_encoder.train()
    text_embedder.train()
    
    return wer, cer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging every 100 steps")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run a small sample for quick GPU test using 73-entry demo dataset",
    )
    parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving the model (in epochs)")

    args = parser.parse_args()
    train(args)
