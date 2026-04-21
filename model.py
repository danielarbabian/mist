import math

from tinygrad import Tensor, nn


class SinusoidalEmbedding:
    def __init__(self, dim: int):
        half = dim // 2
        self.freqs = (-math.log(10000) * Tensor.arange(half) / half).exp()

    def __call__(self, t: Tensor) -> Tensor:
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0)
        return args.sin().cat(args.cos(), dim=-1)


class AdaLN:
    def __init__(self, dim: int):
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 2)

    def __call__(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.proj(cond).unsqueeze(1)
        scale, shift = out.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class SelfAttention:
    def __init__(self, dim: int, n_heads: int):
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def __call__(self, x: Tensor) -> Tensor:
        B, T, C = x.shape[0], x.shape[1], x.shape[2]
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = q.scaled_dot_product_attention(k, v)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class FeedForward:
    def __init__(self, dim: int, mult: int = 4):
        self.w1 = nn.Linear(dim, dim * mult)
        self.w2 = nn.Linear(dim * mult, dim)

    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).gelu())


class TransformerBlock:
    def __init__(self, dim: int, n_heads: int):
        self.attn_norm = AdaLN(dim)
        self.ffn_norm = AdaLN(dim)
        self.attn = SelfAttention(dim, n_heads)
        self.ffn = FeedForward(dim)

    def __call__(self, x: Tensor, cond: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x, cond))
        x = x + self.ffn(self.ffn_norm(x, cond))
        return x


class DiffusionTransformer:
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 256,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.mask_token_id = vocab_size

        self.token_emb = nn.Embedding(vocab_size + 1, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Self-conditioning embedding. Zero-initialised so it's a no-op until
        # the model is trained with --self-conditioning enabled.
        self.self_cond_emb = nn.Embedding(vocab_size + 1, dim)
        self.self_cond_emb.weight = Tensor.zeros(
            vocab_size + 1, dim,
            dtype=self.self_cond_emb.weight.dtype,
            requires_grad=True,
        )

        self.time_emb = SinusoidalEmbedding(dim)
        self.time_w1 = nn.Linear(dim, dim * 4)
        self.time_w2 = nn.Linear(dim * 4, dim)

        self.blocks = [TransformerBlock(dim, n_heads) for _ in range(n_layers)]
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, vocab_size)

    def __call__(
        self, x: Tensor, t: Tensor, self_cond: Tensor | None = None
    ) -> Tensor:
        assert x.shape[1] <= self.max_seq_len, (
            f"seq_len {x.shape[1]} exceeds max_seq_len {self.max_seq_len}"
        )
        h = self.token_emb(x) + self.pos_emb(Tensor.arange(x.shape[1]))
        if self_cond is not None:
            h = h + self.self_cond_emb(self_cond)
        cond = self.time_w2(self.time_w1(self.time_emb(t)).gelu())
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.out_norm(h))
