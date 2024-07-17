import torch
from torch import nn

from einops import repeat, rearrange


class DecoderBlock(nn.Module):
    """
    Causal transformer block
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim, num_heads)
        self.self_attn_norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.fc_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = torch.full(
            (len(x), len(x)),
            -float("Inf"),
            device=x.device,
            dtype=x.dtype,
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm(x + a1)
        a2 = self.fc(a1)
        a2 = self.fc_norm(a1 + a2)

        return a2


class Transformer(nn.Module):
    """
    Decoder only transformer
    """

    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        num_tokens: int,
        seq_len: int,
    ) -> None:
        super().__init__()

        self.token_embeddings = nn.Embedding(num_tokens, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        self.model = nn.Sequential(
            *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, num_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, context_len = x.shape

        token_embeddings = self.token_embeddings(x)

        positions = repeat(
            torch.arange(context_len, device=x.device), "p -> b p", b=batch_size
        )
        position_embeddings = self.position_embeddings(positions)

        embedding = token_embeddings + position_embeddings
        embedding = rearrange(embedding, "b s d -> s b d")

        return self.model(embedding)
