import torch
import argparse
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
import logging
from models.mamba_denoiser import MambaDenoiser
from models.audio_encoder import AudioEncoder
from models.embed import TextEmbedding
from models.masked_diffusion_scheduler import MaskedDiffusionScheduler
from dataset.dataloader import load_data
from utils.consts import PAD, EOS
from utils.metrics import WERMetric
from utils.tokenize import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MaskGitSampler:
    def __init__(self, model, audio_encoder, text_embedder, scheduler, tokenizer, device, mask_token_id, eos_token_id):
        self.model = model
        self.audio_encoder = audio_encoder
        self.text_embedder = text_embedder
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = device
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id

    @torch.inference_mode()
    def generate(self, val_loader, num_steps=20, temperature=1.0, num_batches=None):
        self.model.eval()
        self.audio_encoder.eval()
        self.text_embedder.eval()
        
        metric = WERMetric()
        
        print(f"Starting generation with {num_steps} steps...")
        mask_token_emb = self.text_embedder.tok_emb(
            torch.tensor(self.mask_token_id, device=self.device)
        ).view(1, 1, -1)

        for batch_idx, batch in enumerate(val_loader):
            if num_batches is not None and batch_idx >= num_batches: break
            
            (padded_text_seqs, _, padded_mel_specs, _, _) = batch
            
            batch_size = padded_text_seqs.size(0)
            '''
            In a real inference scenario without ground truth, we would need to use a fixed max length (e.g. 512) 
            and truncate at [EOS] after generation.
            '''
            seq_len = padded_text_seqs.size(1) 
            
            mel_batch = padded_mel_specs.to(self.device).unsqueeze(1)
            
            with autocast('cuda', enabled=(self.device == 'cuda')):
                audio_features = self.audio_encoder(mel_batch)
                
                # creates seq of [MASK] tokens of shape [B, S]
                current_ids = torch.full((batch_size, seq_len), self.mask_token_id, device=self.device, dtype=torch.long)
                x_t, _ = self.text_embedder(current_ids, None)
                
                # Timesteps: 1.0 -> 0.0. We use num_steps iterations
                timesteps = torch.linspace(1, 0, num_steps, device=self.device)
                
                for i, t_val in enumerate(tqdm(timesteps, desc=f"Batch {batch_idx+1} Sampling")):
                    t = torch.full((batch_size,), t_val.item(), device=self.device)
                    logits = self.model(x_t, t, audio_features)
                    
                    probs = F.softmax(logits / temperature, dim=-1)
                    # Using multinomial sampling allows for diversity since argmax is by definition greedy
                    if temperature < 1e-5:
                        sampled_ids = torch.argmax(logits, dim=-1)
                    else:
                        sampled_ids = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, seq_len)
                    confidence = torch.gather(probs, -1, sampled_ids.unsqueeze(-1)).squeeze(-1)
                    
                    
                    if i < num_steps - 1:
                        t_next = timesteps[i+1]
                        
                        # Get masking rate for t_next (e.g. cosine schedule)
                        # We want to know what percentage of tokens should be masked at the NEXT step.
                        # t=1 -> 100% masked, t=0 -> 0% masked
                        mask_ratio = self.scheduler.get_masking_rate(torch.tensor([t_next], device=self.device)).item()
                        num_to_mask = int(mask_ratio * seq_len)
                        
                        # Mask the lowest confidence tokens (MaskGIT strategy)
                        # We keep the (seq_len - num_to_mask) tokens with highest confidence.
                        # So we mask the 'num_to_mask' tokens with lowest confidence.
                        _, mask_indices = torch.topk(confidence, num_to_mask, dim=1, largest=False)
                        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=self.device)
                        mask.scatter_(1, mask_indices, True)
                        
                        # Update x_t for next iteration
                        # Where mask is True -> [MASK] embedding
                        # Where mask is False -> Embedding of sampled_ids (the "kept" tokens)

                        next_ids = sampled_ids.clone()
                        next_ids[mask] = self.mask_token_id
                        x_t, _ = self.text_embedder(next_ids, None)
                        if i % 5 == 0:
                            logger.info(f"Step {i}: Mask Ratio={mask_ratio:.2f}, Num Masked={num_to_mask}, Max Conf={confidence.max().item():.4f}, Min Conf={confidence.min().item():.4f}")
                            sample_dec = self.tokenizer.decode(next_ids[0].tolist(), skip_special_tokens=False)
                            logger.info(f"Step {i} Sample: {sample_dec[:100]}...")
                        
                    else:

                        final_ids = sampled_ids

            predictions = []
            references = []
            
            print("\n" + "="*50)
            for j in range(batch_size):
                pred_seq = final_ids[j].tolist()
      
                if self.eos_token_id in pred_seq:
                    eos_idx = pred_seq.index(self.eos_token_id)
                    pred_seq = pred_seq[:eos_idx]
                
                pred_text = self.tokenizer.decode(pred_seq, skip_special_tokens=True)
                ref_text = self.tokenizer.decode(padded_text_seqs[j].tolist(), skip_special_tokens=True)
                predictions.append(pred_text)
                references.append(ref_text)
                print(f"Ref:  {ref_text}")
                print(f"Pred: {pred_text}")
                print("-" * 20)
            print("="*50 + "\n")
            
            metric.update(predictions, references)
            
        wer = metric.total_wer / metric.count if metric.count > 0 else 0.0
        cer = metric.total_cer / metric.count if metric.count > 0 else 0.0
        print(f"Final WER: {wer:.2f}")
        print(f"Final CER: {cer:.2f}")


def inference(args):
    device = args.device
    tokenizer_path = "tokenizer/tokenizer.json"
    tokenizer, token_ids = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    pad_token_id = token_ids["pad"]
    mask_token_id = token_ids["mask"]
    eos_token_id = token_ids["eos"]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {token_ids}")

    audio_encoder = AudioEncoder(in_channels=1, dropout=0.1).to(device)
    text_embedder = TextEmbedding(
        vocab_size=vocab_size,
        embed_dim=args.d_model,
    ).to(device)
    
    model = MambaDenoiser(
        vocab_size=vocab_size,
        text_dim=args.d_model,
        d_model=args.d_model,
        audio_channels=128,
        n_layers=args.n_layers,
        mask_token_id=mask_token_id,
    ).to(device)

    scheduler = MaskedDiffusionScheduler(
        num_steps=args.num_steps,
        masking_schedule="cosine"
    )
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        def load_state_dict_safe(model, state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)

        load_state_dict_safe(model, checkpoint['model_state_dict'])
        load_state_dict_safe(audio_encoder, checkpoint['audio_encoder_state_dict'])
        load_state_dict_safe(text_embedder, checkpoint['text_embedder_state_dict'])
    else:
        print("Warning: No checkpoint provided. Using random weights.")

    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        audio_encoder = torch.compile(audio_encoder)
        text_embedder = torch.compile(text_embedder)

    print(f"Loading data (split={args.split})...")
    _, val_dl = load_data(
        batch_size=args.batch_size, 
        tokenizer=tokenizer, 
        sample=args.sample,
        num_workers=1,
        val_split=args.split
    )

    sampler = MaskGitSampler(model, audio_encoder, text_embedder, scheduler, tokenizer, device, mask_token_id, eos_token_id)
    sampler.generate(val_dl, num_steps=args.num_steps, temperature=args.temperature, num_batches=args.num_batches)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=None, help="Number of batches to process (default: all)")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=20, help="Number of sampling steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample", action="store_true", help="Use sample dataset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g. test, validation)")
    
    args = parser.parse_args()
    inference(args)
