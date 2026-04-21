import argparse
import csv
import math
import os
import time

from datasets import load_dataset
from tinygrad import Tensor, nn
from tinygrad.nn.state import get_state_dict, safe_save

from config import ENC, VOCAB_SIZE
from diffusion import compute_loss, forward_process
from model import DiffusionTransformer


def clip_grad_norm(parameters: list[Tensor], max_norm: float = 1.0) -> Tensor:
    """Clip gradient norm across all parameters. Returns the total norm."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return Tensor(0.0)
    total_norm_sq = Tensor(0.0)
    for g in grads:
        total_norm_sq = total_norm_sq + g.float().square().sum()
    total_norm = total_norm_sq.sqrt()
    clip_coef = (max_norm / (total_norm + 1e-6)).minimum(1.0)
    for p in parameters:
        if p.grad is not None:
            p.grad = p.grad * clip_coef
    return total_norm


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    schedule: str,
) -> float:
    """Linear warmup then cosine decay or constant. step is 1-indexed."""
    if step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_ema_state(model) -> dict[str, Tensor]:
    """Clone current model state into fresh Tensors for EMA tracking."""
    return {
        k: Tensor(v.numpy(), dtype=v.dtype, requires_grad=False)
        for k, v in get_state_dict(model).items()
    }


def update_ema(ema_state: dict[str, Tensor], model, decay: float) -> None:
    """In-place EMA update: ema = decay * ema + (1 - decay) * model."""
    updates = [
        ema_state[k].assign(ema_state[k] * decay + v.detach() * (1 - decay))
        for k, v in get_state_dict(model).items()
    ]
    Tensor.realize(*updates)


def _encode_padded(story: str, enc, seq_len: int, pad_token: int) -> tuple[list[int], list[bool]]:
    ids = enc.encode(story)[:seq_len]
    real_len = len(ids)
    if real_len < seq_len:
        ids = ids + [pad_token] * (seq_len - real_len)
    return ids, [True] * real_len + [False] * (seq_len - real_len)


def make_batch(
    ds, iterator, enc, batch_size: int, seq_len: int, pad_token: int, epoch: int, seed: int
):
    """Build a batch of token sequences. Returns (tokens, padding_mask, iterator, epoch).

    Restarts the dataset iterator on exhaustion for multi-epoch training.
    padding_mask is True for real tokens, False for padding.
    """
    tokens = []
    pad_masks = []
    while len(tokens) < batch_size:
        try:
            story = next(iterator)["text"]
        except StopIteration:
            epoch += 1
            ds = ds.shuffle(seed=seed + epoch)
            iterator = iter(ds)
            story = next(iterator)["text"]
        ids, pm = _encode_padded(story, enc, seq_len, pad_token)
        tokens.append(ids)
        pad_masks.append(pm)
    return Tensor(tokens), Tensor(pad_masks), iterator, epoch


def make_val_batches(
    stories: list[str], enc, batch_size: int, seq_len: int, pad_token: int
) -> list[tuple[Tensor, Tensor]]:
    """Pre-tokenise a fixed list of stories into (tokens, pad_mask) batches.

    Drops any incomplete tail batch so every batch has exactly batch_size rows.
    """
    batches: list[tuple[Tensor, Tensor]] = []
    buf_tokens: list[list[int]] = []
    buf_masks: list[list[bool]] = []
    for story in stories:
        ids, pm = _encode_padded(story, enc, seq_len, pad_token)
        buf_tokens.append(ids)
        buf_masks.append(pm)
        if len(buf_tokens) == batch_size:
            batches.append((Tensor(buf_tokens), Tensor(buf_masks)))
            buf_tokens, buf_masks = [], []
    return batches


def compute_val_loss(
    model, val_batches: list[tuple[Tensor, Tensor]], mask_token_id: int, val_seed: int
) -> float:
    """Forward-only eval across fixed batches.

    Re-seeds the RNG before eval so val loss is reproducible across checkpoints.
    Side effect: advances the global RNG, so training sees a deterministic but
    different sequence than a run without val eval.
    """
    was_training = Tensor.training
    Tensor.training = False
    try:
        Tensor.manual_seed(val_seed)
        total = 0.0
        n = 0
        for x0, pad_mask in val_batches:
            B = x0.shape[0]
            t = Tensor.rand(B)
            xt, mask = forward_process(x0, t, mask_token_id=mask_token_id)
            logits = model(xt, t)
            loss = compute_loss(logits, x0, mask, pad_mask)
            total += loss.item()
            n += 1
        return total / max(n, 1)
    finally:
        Tensor.training = was_training


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    Tensor.manual_seed(args.seed)

    enc = ENC
    ds = load_dataset("roneneldan/TinyStories", split="train")
    ds = ds.shuffle(seed=args.seed)
    iterator = iter(ds)

    val_batches: list[tuple[Tensor, Tensor]] = []
    if args.val_every > 0 and args.val_size > 0:
        val_ds = load_dataset("roneneldan/TinyStories", split="validation")
        val_stories = [val_ds[i]["text"] for i in range(min(args.val_size, len(val_ds)))]
        val_batches = make_val_batches(
            val_stories, enc, args.batch_size, args.seq_len, enc.eot_token
        )
        print(
            f"Val: {len(val_batches)} batches ({len(val_batches) * args.batch_size} sequences), every {args.val_every} steps"
        )

    model = DiffusionTransformer(
        vocab_size=VOCAB_SIZE,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )
    params = nn.state.get_parameters(model)
    optim = nn.optim.Adam(params, lr=args.lr)

    ema_state = make_ema_state(model) if args.ema_decay > 0 else None

    param_count = sum(p.numel() for p in params)
    print(f"Model params: {param_count / 1e6:.1f}M")
    print(
        f"Training for {args.steps} steps, batch_size={args.batch_size}, seq_len={args.seq_len}"
    )
    lr_tail = f"cosine decay to {args.min_lr}" if args.lr_schedule == "cosine" else "constant"
    print(f"LR: {args.warmup_steps}-step warmup → {lr_tail}, max_grad_norm={args.max_grad_norm}")
    if ema_state is not None:
        print(f"EMA enabled, decay={args.ema_decay}")

    log_path = os.path.join(args.checkpoint_dir, "train_log.csv")
    log_file = None
    val_seed = args.seed + 1_000_000

    epoch = 0
    Tensor.training = True
    try:
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            ["step", "loss", "val_loss", "grad_norm", "lr", "tokens_per_sec", "epoch", "dt"]
        )
        for step in range(1, args.steps + 1):
            t0 = time.monotonic()

            lr = get_lr(
                step, args.warmup_steps, args.steps, args.lr, args.min_lr, args.lr_schedule
            )
            optim.lr.assign(Tensor([lr], dtype=optim.lr.dtype)).realize()

            x0, pad_mask, iterator, epoch = make_batch(
                ds, iterator, enc, args.batch_size, args.seq_len, enc.eot_token, epoch, args.seed
            )
            t = Tensor.rand(args.batch_size)
            xt, mask = forward_process(x0, t, mask_token_id=VOCAB_SIZE)
            logits = model(xt, t)
            loss = compute_loss(logits, x0, mask, pad_mask)

            optim.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm(params, max_norm=args.max_grad_norm)
            optim.step()

            if ema_state is not None:
                update_ema(ema_state, model, args.ema_decay)

            elapsed = time.monotonic() - t0
            loss_val = loss.item()
            grad_norm_val = grad_norm.item()
            tokens_per_sec = args.batch_size * args.seq_len / max(elapsed, 1e-9)

            val_loss_str = ""
            if val_batches and step % args.val_every == 0:
                val_loss = compute_val_loss(model, val_batches, VOCAB_SIZE, val_seed)
                val_loss_str = f"{val_loss:.4f}"

            log_writer.writerow(
                [
                    step,
                    f"{loss_val:.4f}",
                    val_loss_str,
                    f"{grad_norm_val:.4f}",
                    f"{lr:.2e}",
                    f"{tokens_per_sec:.0f}",
                    epoch,
                    f"{elapsed:.2f}",
                ]
            )

            if step % args.log_every == 0 or val_loss_str:
                msg = (
                    f"step={step:>5d}  loss={loss_val:.4f}  "
                    f"grad={grad_norm_val:.2f}  lr={lr:.2e}  "
                    f"tps={tokens_per_sec:.0f}  dt={elapsed:.2f}s"
                )
                if val_loss_str:
                    msg += f"  val={val_loss_str}"
                print(msg)
                log_file.flush()

            if step % args.save_every == 0:
                _save_checkpoint(args.checkpoint_dir, step, model, ema_state)
    finally:
        Tensor.training = False
        if log_file is not None:
            log_file.close()

    _save_checkpoint(args.checkpoint_dir, args.steps, model, ema_state)
    print("Training complete.")


def _save_checkpoint(
    checkpoint_dir: str, step: int, model, ema_state: dict[str, Tensor] | None
) -> None:
    path = os.path.join(checkpoint_dir, f"model_{step}.safetensors")
    safe_save(get_state_dict(model), path)
    print(f"Saved checkpoint: {path}")
    if ema_state is not None:
        ema_path = os.path.join(checkpoint_dir, f"model_{step}_ema.safetensors")
        safe_save(ema_state, ema_path)
        print(f"Saved EMA checkpoint: {ema_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a discrete diffusion transformer on TinyStories"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
