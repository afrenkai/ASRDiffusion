import torch
from typing import List, Dict


def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # do this one on leetcode its actually a fun dp problem imo
    # https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    wer = dp[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer * 100


def calculate_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    dp = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        dp[i][0] = i
    for j in range(len(hyp_chars) + 1):
        dp[0][j] = j
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    cer = dp[len(ref_chars)][len(hyp_chars)] / len(ref_chars)
    return cer * 100


class WERMetric:
    def __init__(self) -> None:
        self.total_wer = 0.0
        self.total_cer = 0.0
        self.count = 0

    def update(self, predictions, references) -> None:
        if isinstance(predictions, str) and isinstance(references, str):
            preds = [predictions]
            refs = [references]
        else:
            preds = list(predictions)
            refs = list(references)
        if len(preds) != len(refs):
            raise ValueError("predictions and references must have the same length")
        for p, r in zip(preds, refs):
            wer = calculate_wer(r, p)
            cer = calculate_cer(r, p)
            self.total_wer += wer
            self.total_cer += cer
            self.count += 1

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"wer": 0.0, "cer": 0.0, "count": 0}
        return {
            "wer": self.total_wer / self.count,
            "cer": self.total_cer / self.count,
            "count": self.count,
        }


def evaluate_batch(
    model,
    sampler,
    audio_features: torch.Tensor,
    reference_texts: List[str],
    text_embedding_model,
    tokenizer,
    device: str = "cuda",
    seq_len: int | None = None,
) -> tuple[List[float], List[str]]:
    batch_size = audio_features.size(0)
    try:
        text_dim = getattr(text_embedding_model, "embed_dim", None)
        if text_dim is None:
            text_dim = text_embedding_model.tok_emb.weight.shape[1]
    except Exception:
        text_dim = text_embedding_model.tok_emb.weight.shape[1]
    if seq_len is None:
        try:
            max_pos = text_embedding_model.pos_enc.shape[0]
            seq_len = min(64, max_pos)
        except Exception:
            seq_len = 64
    shape = (batch_size, seq_len, text_dim)
    audio_features = audio_features.to(device)
    emb = sampler.sample(
        model=model,
        shape=shape,
        audio_features=audio_features,
        device=device,
    )
    # this is NOT optimal, there's a better way to do this, but its getting late in the day
    with torch.no_grad():
        token_emb = text_embedding_model.tok_emb.weight.to(emb.device)
        distances = torch.cdist(emb, token_emb)
        token_ids = distances.argmin(dim=-1)
    texts = []
    scores = []
    for b in range(batch_size):
        ids = token_ids[b].tolist()
        try:
            text = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            text = ""

        texts.append(text)

        mean_min_dist = distances[b].min(dim=-1).values.mean().item()
        score = float(1.0 / (1.0 + mean_min_dist))
        scores.append(score)
    return scores, texts


if __name__ == "__main__":
    ref = "shabbat shalom"
    hyp = "shabbas shalom"
    print(f"WER: {calculate_wer(ref, hyp):.2f}%")
    print(f"CER: {calculate_cer(ref, hyp):.2f}%")

