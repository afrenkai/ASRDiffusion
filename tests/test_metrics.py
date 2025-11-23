import torch
import pytest

from utils.metrics import (
    calculate_wer,
    calculate_cer,
    WERMetric,
    evaluate_batch,
)
from models.embed import TextEmbedding


class DummyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        try:
            return " ".join(str(int(i)) for i in ids)
        except Exception:
            return ""


class DummySampler:
    def __init__(self, seed=0):
        self.seed = seed

    def sample(self, model, shape, audio_features, device="cpu", mask=None):
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        return torch.randn(*shape, device=device, generator=gen)


def test_calculate_wer_basic():
    ref = "this is a test"
    hyp = "this is test"
    wer = calculate_wer(ref, hyp)
    assert pytest.approx(wer, rel=1e-3) == 25.0


def test_calculate_cer_basic():
    ref = "abc"
    hyp = "ac"
    cer = calculate_cer(ref, hyp)
    assert pytest.approx(cer, rel=1e-3) == (1.0 / 3.0) * 100


def test_WERMetric_updates_and_compute():
    m = WERMetric()
    m.update("hello world", "hello world")
    assert m.compute()["wer"] == 0.0

    preds = ["a b c", "d e f"]
    refs = ["a b c", "d x f"]
    m.update(preds, refs)
    res = m.compute()
    assert res["count"] == 3
    expected = (0.0 + 0.0 + (1.0 / 3.0) * 100) / 3.0
    assert pytest.approx(res["wer"], rel=1e-3) == expected


def test_evaluate_batch_returns_texts_and_scores():
    batch_size = 2
    audio_features = torch.randn(batch_size, 10)
    text_model = TextEmbedding(vocab_size=50, embed_dim=32)
    tokenizer = DummyTokenizer()
    sampler = DummySampler(seed=42)

    scores, texts = evaluate_batch(
        model=None,
        sampler=sampler,
        audio_features=audio_features,
        reference_texts=["ref1", "ref2"],
        text_embedding_model=text_model,
        tokenizer=tokenizer,
        device="cpu",
        seq_len=8,
    )

    assert isinstance(scores, list) and isinstance(texts, list)
    assert len(scores) == batch_size and len(texts) == batch_size
    for s in scores:
        assert 0.0 <= s <= 1.0
    for t in texts:
        assert isinstance(t, str)
