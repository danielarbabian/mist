import math
from typing import Protocol

from tinygrad import Tensor


class Denoiser(Protocol):
    vocab_size: int

    def __call__(self, x: Tensor, t: Tensor) -> Tensor: ...


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


def compute_loss(logits: Tensor, x0: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy loss on masked positions only.

    Args:
        logits: Model output, shape (B, T, vocab_size)
        x0: Clean token IDs, shape (B, T)
        mask: Boolean mask, shape (B, T) — True where masked

    Returns:
        Scalar loss averaged over masked positions.
    """
    B, T, V = logits.shape[0], logits.shape[1], logits.shape[2]
    per_token_loss = logits.reshape(B * T, V).cross_entropy(x0.reshape(B * T), reduction="none")
    per_token_loss = per_token_loss.reshape(B, T)
    masked_loss = (per_token_loss * mask).sum() / mask.sum().maximum(1)
    return masked_loss


def sample(
    model: Denoiser,
    seq_len: int,
    num_steps: int,
    vocab_size: int,
    batch_size: int = 1,
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

    Returns:
        Generated token IDs, shape (batch_size, seq_len)
    """
    mask_token_id = vocab_size
    xt = Tensor.full((batch_size, seq_len), mask_token_id)

    for i in range(num_steps, 0, -1):
        t = i / num_steps
        t_prev = (i - 1) / num_steps

        t_tensor = Tensor.full((batch_size,), t)
        logits = model(xt, t_tensor)
        sampled = logits.argmax(axis=-1)

        is_masked = xt == mask_token_id

        if i == 1:
            # Final step: unmask everything remaining
            xt = Tensor.where(is_masked, sampled, xt)
        else:
            alpha_t = noise_schedule(Tensor([t])).item()
            alpha_prev = noise_schedule(Tensor([t_prev])).item()
            eps = 1e-8
            unmask_prob = (alpha_prev - alpha_t) / (1 - alpha_t + eps)
            unmask = is_masked * (Tensor.rand(batch_size, seq_len) < unmask_prob)
            xt = Tensor.where(unmask, sampled, xt)

    return xt
