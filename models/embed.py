import torch
import torch.nn as nn
import math


class TextEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int = 67, embed_dim: int = 100, max_seq_len: int = 512
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        print("emb work")
        self.register_buffer(
            "pos_enc", self._create_positional_encoding(max_seq_len, embed_dim)
        )

    def _create_positional_encoding(self, max_len, d_model):
        # sinusoidal pe from vaswani et al
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10.0000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self, tok: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq = tok.size()
        tok_emb = self.tok_emb(tok)
        pos_emb = self.pos_enc[:seq, :].unsqueeze(0)
        emb = tok_emb + pos_emb
        if mask is not None:
            emb = emb * mask.unsqueeze(-1)

        return emb, mask


if __name__ == "__main__":
    mod = TextEmbedding()
    tok = torch.randint(3, 5, (4, 16))

    emb, mask = mod.forward(tok, None)
    print(emb.size())
