import torch
import torch.nn.functional as F
from models.mamba_denoiser import MambaDenoiser
from models.pipeline import load_librispeech_samples, create_mels_batches, encode
from models.audio_encoder import AudioEncoder
from transformers import AutoTokenizer
import argparse
import math


@torch.no_grad()
def noise_schedule(t: torch.Tensor, eps=1e-5):
    """
    Cosine noise schedule from improved DDPM
    
    Args:
        t: timesteps of shape [B] in range [0, 1]
        eps: small constant for numerical stability
    
    Returns:
        alpha_t: signal coefficients of shape [B]
        sigma_t: noise coefficients of shape [B]
    """
    s = eps
    f_t = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    f_0 = math.cos(s / (1 + s) * (math.pi / 2)) ** 2
    
    alpha_t_squared = f_t / f_0
    alpha_t = torch.sqrt(alpha_t_squared)
    sigma_t = torch.sqrt(1 - alpha_t_squared)
    
    return alpha_t, sigma_t


@torch.no_grad()
def reverse_diffusion_sample(
    model, 
    audio_features, 
    num_steps=50, 
    seq_len=64,
    device="cuda",
    temperature=1.0,
    cfg_scale=0.0
):
    """
    Ancestral sampling from the reverse diffusion process using score-based SDE
    
    Args:
        model: trained MambaDenoiser
        audio_features: encoded audio of shape [B x C x T' x F']
        num_steps: number of reverse diffusion steps
        seq_len: target sequence length
        device: device to run on
        temperature: sampling temperature (1.0 = standard, <1.0 = sharper)
        cfg_scale: classifier-free guidance scale (0.0 = no guidance)
    
    Returns:
        x_0: predicted clean text embeddings of shape [B x S x E]
    """
    model.eval()
    batch_size = audio_features.size(0)

    x_t = torch.randn(batch_size, seq_len, model.text_dim, device=device)
    
    dt = 1.0 / num_steps
    
    for i in reversed(range(num_steps)):
        t_continuous = torch.full((batch_size,), i / num_steps, device=device)

        score = model(
            x_t=x_t, 
            t=t_continuous, 
            audio_features=audio_features, 
            mask=None,
            corrupt_mask=None
        )
        
     
        alpha_t, sigma_t = noise_schedule(t_continuous)
        alpha_t = alpha_t.view(-1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1)
        
        # Euler-Maruyama step for reverse-time SDE
        # dx = [sigma_t^2 * score] dt + sigma_t * sqrt(dt) * dW
        drift = sigma_t**2 * score
        
        if i > 0:
            # Add stochastic term
            diffusion_coeff = sigma_t * torch.sqrt(torch.tensor(dt, device=device))
            z = torch.randn_like(x_t) * temperature
            x_t = x_t + drift * dt + diffusion_coeff * z
        else:
            # Final step - deterministic
            x_t = x_t + drift * dt
    
    return x_t


@torch.no_grad()
def ddim_sample(
    model,
    audio_features,
    num_steps=50,
    seq_len=64,
    device="cuda",
    eta=0.0
):
    """
    DDIM sampling for faster deterministic generation
    
    Args:
        model: trained MambaDenoiser
        audio_features: encoded audio of shape [B x C x T' x F']
        num_steps: number of reverse diffusion steps
        seq_len: target sequence length
        device: device to run on
        eta: stochasticity parameter (0.0 = deterministic DDIM, 1.0 = DDPM)
    
    Returns:
        x_0: predicted clean text embeddings of shape [B x S x E]
    """
    model.eval()
    batch_size = audio_features.size(0)
    
    x_t = torch.randn(batch_size, seq_len, model.text_dim, device=device)
    
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    
    for i in range(num_steps):
        t = timesteps[i].expand(batch_size)
        t_next = timesteps[i + 1].expand(batch_size)

        score = model(
            x_t=x_t,
            t=t,
            audio_features=audio_features,
            mask=None,
            corrupt_mask=None
        )
        
        alpha_t, sigma_t = noise_schedule(t)
        alpha_next, sigma_next = noise_schedule(t_next)
        
        alpha_t = alpha_t.view(-1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1)
        alpha_next = alpha_next.view(-1, 1, 1)
        sigma_next = sigma_next.view(-1, 1, 1)
        

        x_0_pred = (x_t + sigma_t**2 * score) / alpha_t
        
        if i < num_steps - 1:
            noise_factor = eta * torch.sqrt(
                (sigma_next**2 / sigma_t**2) * (1 - alpha_t**2 / alpha_next**2)
            )
            x_t = alpha_next * x_0_pred + torch.sqrt(sigma_next**2 - noise_factor**2) * (
                (x_t - alpha_t * x_0_pred) / sigma_t
            )
            
            if eta > 0:
                x_t = x_t + noise_factor * torch.randn_like(x_t)
        else:
            x_t = x_0_pred
    
    return x_t


def decode_embeddings(embeddings, tokenizer, vocab_size=50000):
    """
    Decode text embeddings to text (placeholder - needs proper implementation)
    
    Args:
        embeddings: text embeddings of shape [B x S x E]
        tokenizer: tokenizer to decode with
        vocab_size: vocabulary size
    
    Returns:
        texts: list of decoded text strings
    """
    # TODO: Implement proper decoding
    # Options:
    # 1. Nearest neighbor in embedding space
    # 2. Trained decoder head
    # 3. CTC-style decoding
    
    batch_size, seq_len, embed_dim = embeddings.shape
    
    # Placeholder: random decoding
    print("Warning: Using placeholder decoding. Implement proper decoder!")
    texts = [f"[DECODED_TEXT_{i}]" for i in range(batch_size)]
    
    return texts


def inference(args):
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
        dropout=0.0,  
        device=device
    ).to(device)
    

    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint_path}")
        print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    else:
        print("Warning: No checkpoint loaded. Using random weights.")
    
    model.eval()

    audio_encoder = AudioEncoder(in_channels=1, dropout=0.0).to(device)

    print(f"Loading {args.num_samples} samples...")
    samples, texts, srs = load_librispeech_samples(
        num_samples=args.num_samples,
        split=args.split
    )
    
    print("Ground truth texts:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")

    mel_batches = create_mels_batches(samples)
    encoded_audio = encode(audio_encoder, mel_batches, device=device)

    print(f"\nRunning {args.sampling_method} sampling with {args.num_steps} steps...")
    
    if args.sampling_method == "ancestral":
        predicted_embeddings = reverse_diffusion_sample(
            model=model,
            audio_features=encoded_audio,
            num_steps=args.num_steps,
            seq_len=args.seq_len,
            device=device,
            temperature=args.temperature
        )
    elif args.sampling_method == "ddim":
        predicted_embeddings = ddim_sample(
            model=model,
            audio_features=encoded_audio,
            num_steps=args.num_steps,
            seq_len=args.seq_len,
            device=device,
            eta=args.eta
        )
    else:
        raise ValueError(f"Unknown sampling method: {args.sampling_method}")
    
    print(f"Predicted embeddings shape: {predicted_embeddings.shape}")
  
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    predicted_texts = decode_embeddings(predicted_embeddings, tokenizer)
    
    print("\nPredicted texts:")
    for i, text in enumerate(predicted_texts):
        print(f"  {i+1}. {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with Mamba Denoiser for ASR')

    parser.add_argument('--text_dim', type=int, default=128)
    parser.add_argument('--audio_channels', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--headdim', type=int, default=64)
    parser.add_argument('--ngroups', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=4)

    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--split', type=str, default='test.clean')
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--sampling_method', type=str, default='ddim', 
                        choices=['ancestral', 'ddim'])
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--tokenizer_name', type=str, default='google/gemma-2-2b-it')
    
    args = parser.parse_args()
    
    inference(args)
