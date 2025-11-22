import torch
from models.mamba_denoiser import MambaDenoiser
from models.audio_encoder import AudioEncoder
from models.pipeline import load_librispeech_samples, create_mels_batches, encode
from inference import ddim_sample, decode_embeddings
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import jiwer
import numpy as np


def compute_wer(references, hypotheses):
    """
    Compute Word Error Rate
    
    Args:
        references: list of reference texts
        hypotheses: list of hypothesis texts
    
    Returns:
        wer: word error rate (0.0 to 1.0+)
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])
    
    wer = jiwer.wer(
        references,
        hypotheses,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    
    return wer


def compute_cer(references, hypotheses):
    """
    Compute Character Error Rate
    
    Args:
        references: list of reference texts
        hypotheses: list of hypothesis texts
    
    Returns:
        cer: character error rate (0.0 to 1.0+)
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=False),
        jiwer.ReduceToListOfListOfChars()
    ])
    
    cer = jiwer.wer(
        references,
        hypotheses,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    
    return cer


def compute_embedding_distance(pred_embeddings, true_embeddings):
    """
    Compute various distance metrics between predicted and true embeddings
    
    Args:
        pred_embeddings: predicted embeddings [B x S x E]
        true_embeddings: ground truth embeddings [B x S x E]
    
    Returns:
        dict of distance metrics
    """
    l2_dist = torch.norm(pred_embeddings - true_embeddings, p=2, dim=-1).mean().item()

    pred_norm = torch.nn.functional.normalize(pred_embeddings, dim=-1)
    true_norm = torch.nn.functional.normalize(true_embeddings, dim=-1)
    cosine_sim = (pred_norm * true_norm).sum(dim=-1).mean().item()
    
    return {
        'l2_distance': l2_dist,
        'cosine_similarity': cosine_sim
    }


@torch.no_grad()
def evaluate(args):
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    all_wer = []
    all_cer = []
    all_l2_distances = []
    all_cosine_similarities = []
    
    all_references = []
    all_hypotheses = []
    
    print(f"Evaluating on {args.split} split...")
    print(f"Total samples to evaluate: {args.num_samples}")
    num_batches = args.num_samples // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
        samples, texts, srs = load_librispeech_samples(
            num_samples=args.batch_size,
            split=args.split
        )
        
        all_references.extend(texts)
        mel_batches = create_mels_batches(samples)
        encoded_audio = encode(audio_encoder, mel_batches, device=device)
        predicted_embeddings = ddim_sample(
            model=model,
            audio_features=encoded_audio,
            num_steps=args.num_steps,
            seq_len=args.seq_len,
            device=device,
            eta=args.eta
        )
        hypotheses = decode_embeddings(predicted_embeddings, tokenizer)
        all_hypotheses.extend(hypotheses)
        if args.compute_embedding_metrics:
            from models.pipeline import pipeline
            t = torch.zeros(args.batch_size, device=device)
            _, true_embeddings, _ = pipeline(
                samples, texts, t, tokenizer,
                use_char_level=args.use_char_level,
                device=device
            )
            
            min_len = min(predicted_embeddings.size(1), true_embeddings.size(1))
            pred_crop = predicted_embeddings[:, :min_len, :]
            true_crop = true_embeddings[:, :min_len, :]
            
            metrics = compute_embedding_distance(pred_crop, true_crop)
            all_l2_distances.append(metrics['l2_distance'])
            all_cosine_similarities.append(metrics['cosine_similarity'])

    wer = compute_wer(all_references, all_hypotheses)
    cer = compute_cer(all_references, all_hypotheses)
    
    print(f"Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
    
    if args.compute_embedding_metrics:
        avg_l2 = np.mean(all_l2_distances)
        avg_cosine = np.mean(all_cosine_similarities)
        print(f"Average L2 Distance: {avg_l2:.4f}")
        print(f"Average Cosine Similarity: {avg_cosine:.4f}")
    

 
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(f"WER: {wer:.4f}\n")
            f.write(f"CER: {cer:.4f}\n")
            if args.compute_embedding_metrics:
                f.write(f"L2 Distance: {avg_l2:.4f}\n")
                f.write(f"Cosine Similarity: {avg_cosine:.4f}\n")
            for ref, hyp in zip(all_references, all_hypotheses):
                f.write(f"REF: {ref}\n")
                f.write(f"HYP: {hyp}\n")
                f.write("-"*50 + "\n")
        print(f"\nResults saved to {args.output_file}")
    
    return {
        'wer': wer,
        'cer': cer,
        'l2_distance': avg_l2 if args.compute_embedding_metrics else None,
        'cosine_similarity': avg_cosine if args.compute_embedding_metrics else None
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Mamba Denoiser for ASR')
    parser.add_argument('--text_dim', type=int, default=128)
    parser.add_argument('--audio_channels', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_state', type=int, default=64)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--headdim', type=int, default=64)
    parser.add_argument('--ngroups', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test.clean')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--tokenizer_name', type=str, default='google/gemma-2-2b-it')
    parser.add_argument('--use_char_level', action='store_true')
    parser.add_argument('--compute_embedding_metrics', action='store_true')
    parser.add_argument('--show_examples', action='store_true')
    parser.add_argument('--output_file', type=str, default=None)
    
    args = parser.parse_args()
    
    evaluate(args)
