from models.embed import TextEmbedding
import torch


def test_embedding(
    model=TextEmbedding,
    low=3,
    high=5,
    size: tuple[int, int] = (4, 6),
    embed_dim: int = 100,
):
    tok = torch.randint(low, high, size=size)
    emb, _ = model().forward(tok=tok, mask=None)
    assert emb.size() == torch.Size([size[0], size[1], embed_dim])
