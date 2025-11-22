import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.mamba_denoiser import MambaDenoiser
from models.pipeline import pipeline, load_librispeech_samples
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import os


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = MambaDenoiser(
        text_dim=args.text_dim,
        audio_channels=args.audio_channels,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        headdim=args.headdim,
        ngroups=args.ngroups,
        n_layers=args.n_layers,
        dropout=args.dropout,
        device=device
    ).to(device)
    

    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    total_steps = args.num_epochs * args.steps_per_epoch
    warmup_steps = args.warmup_steps
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + torch.cos(torch.tensor((step - warmup_steps) / (total_steps - warmup_steps) * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        valid_steps = 0
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx in pbar:

            samples, texts, srs = load_librispeech_samples(
                num_samples=args.batch_size, 
                split="train.clean.100"
            )

            t = torch.rand(args.batch_size, device=device)

            encoded_audio, text_embeddings, diffusion_feats = pipeline(
                samples, texts, t, tokenizer, 
                use_char_level=args.use_char_level, 
                device=device
            )
            
            x_t, eps_true, alpha_t, sigma_t = diffusion_feats

            mask = torch.ones(args.batch_size, x_t.size(1), device=device)
            
            score_pred = model(
                x_t=x_t,
                t=t,
                audio_features=encoded_audio,
                mask=mask,
                corrupt_mask=None
            )

            loss = model.compute_score_loss(
                score_pred=score_pred,
                x_t=x_t,
                x_0=text_embeddings,
                sigma_t=sigma_t,
                mask=mask
            )
            
            # TODO: find out why there are NaNs
            if torch.isnan(loss) or torch.isinf(loss):
                pbar.set_postfix({'loss': 'NaN/Inf', 'lr': scheduler.get_last_lr()[0]})
                continue

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            valid_steps += 1
            step += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/valid_steps:.4f}',
                'grad_norm': f'{grad_norm:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

            if step % args.save_every == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_step_{step}.pt')
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'args': vars(args)
                }, checkpoint_path)
                print(f"\nCheckpoint saved to {checkpoint_path}")
        
        avg_loss = epoch_loss / valid_steps if valid_steps > 0 else float('inf')
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Mamba Denoiser for ASR')
    parser.add_argument('--text_dim', type=int, default=128)
    parser.add_argument('--audio_channels', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--headdim', type=int, default=64)
    parser.add_argument('--ngroups', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--tokenizer_name', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--use_char_level', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=1000)
    
    args = parser.parse_args()
    
    train(args)
