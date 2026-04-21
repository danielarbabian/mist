import math
from collections.abc import Callable
from typing import Protocol

from tinygrad import Tensor


class Denoiser(Protocol):
    vocab_size: int

    def __call__(self, x: Tensor, t: Tensor, self_cond: Tensor | None = None) -> Tensor: ...


def noise_schedule(t: Tensor) -> Tensor:
    """Cosine noise schedule. Returns alpha(t) = probability token is kept."""
    return (t * math.pi / 2).cos() ** 2


def forward_process(x0: Tensor, t: Tensor, mask_token_id: int) -> tuple[Tensor, Tensor]:
    """Corrupt clean tokens by masking based on timestep.

    Args:
        x0: Clean token IDs, shape (B, T)
        t: Timesteps in [0, 1], shape (B,)
        mask_token_id: ID to use for masked positions

    Returns:
        xt: Corrupted tokens, shape (B, T)
        mask: Boolean mask of corrupted positions, shape (B, T)
    """
    alpha = noise_schedule(t).unsqueeze(1)  # (B, 1)
    mask = Tensor.rand(*x0.shape) > alpha  # True where masked
    xt = Tensor.where(mask, mask_token_id, x0)
    return xt, mask


def compute_loss(
    logits: Tensor, x0: Tensor, mask: Tensor, pad_mask: Tensor | None = None
) -> Tensor:
    """Cross-entropy loss on masked positions only, excluding padding.

    Args:
        logits: Model output, shape (B, T, vocab_size)
        x0: Clean token IDs, shape (B, T)
        mask: Boolean mask, shape (B, T) — True where masked by diffusion
        pad_mask: Boolean mask, shape (B, T) — True for real tokens, False for padding.
                  If None, all positions are treated as real.

    Returns:
        Scalar loss averaged over masked, non-padding positions.
    """
    B, T, V = logits.shape[0], logits.shape[1], logits.shape[2]
    per_token_loss = logits.reshape(B * T, V).cross_entropy(x0.reshape(B * T), reduction="none")
    per_token_loss = per_token_loss.reshape(B, T)
    effective_mask = mask * pad_mask if pad_mask is not None else mask
    masked_loss = (per_token_loss * effective_mask).sum() / effective_mask.sum().maximum(1)
    return masked_loss


def _sample_tokens(logits: Tensor, temperature: float) -> Tensor:
    """Sample token IDs from logits using temperature scaling + Gumbel-max trick."""
    if temperature < 1e-8:
        return logits.argmax(axis=-1)
    u = Tensor.rand(*logits.shape).clip(1e-20, 1.0 - 1e-7)
    gumbel_noise = -(-(u.log())).log()
    return ((logits / temperature) + gumbel_noise).argmax(axis=-1)


def sample(
    model: Denoiser,
    seq_len: int,
    num_steps: int,
    vocab_size: int,
    batch_size: int = 1,
    temperature: float = 0.8,
    self_cond: bool = False,
    on_step: Callable[[int, int, Tensor, Tensor], None] | None = None,
) -> Tensor:
    """Generate text by iterative unmasking.

    Starts from fully masked sequence and progressively reveals tokens
    over num_steps denoising steps.

    Args:
        model: DiffusionTransformer instance
        seq_len: Length of sequence to generate
        num_steps: Number of denoising steps
        vocab_size: Vocabulary size (mask_token_id = vocab_size)
        batch_size: Number of sequences to generate
        temperature: Sampling temperature (0 = greedy, higher = more random)
        self_cond: If True, feed the previous step's argmax prediction back
                   as a self-conditioning signal. Requires a model trained with
                   --self-conditioning; has no effect on models trained without.
        on_step: Optional callback(step, num_steps, xt, sampled) called after each step

    Returns:
        Generated token IDs, shape (batch_size, seq_len)
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    mask_token_id = vocab_size
    xt = Tensor.full((batch_size, seq_len), mask_token_id)
    sc_pred: Tensor | None = None

    for i in range(num_steps, 0, -1):
        t = i / num_steps
        t_prev = (i - 1) / num_steps

        t_tensor = Tensor.full((batch_size,), t)
        logits = model(xt, t_tensor, sc_pred if self_cond else None)
        sampled = _sample_tokens(logits, temperature)
        if self_cond:
            sc_pred = logits.argmax(axis=-1).detach()

        is_masked = xt == mask_token_id

        if i == 1:
            xt = Tensor.where(is_masked, sampled, xt)
        else:
            alpha_t = noise_schedule(Tensor([t])).item()
            alpha_prev = noise_schedule(Tensor([t_prev])).item()
            eps = 1e-8
            unmask_prob = (alpha_prev - alpha_t) / (1 - alpha_t + eps)
            unmask = is_masked * (Tensor.rand(batch_size, seq_len) < unmask_prob)
            xt = Tensor.where(unmask, sampled, xt)

        if on_step is not None:
            on_step(num_steps - i + 1, num_steps, xt, sampled)

    return xt
