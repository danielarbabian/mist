import argparse
import csv
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


def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup then constant LR. step is 1-indexed."""
    if step <= warmup_steps:
        return base_lr * step / warmup_steps
    return base_lr


def make_batch(
    ds, iterator, enc, batch_size: int, seq_len: int, pad_token: int
):
    """Build a batch of token sequences. Returns (tokens, padding_mask).

    Restarts the dataset iterator on exhaustion for multi-epoch training.
    padding_mask is True for real tokens, False for padding.
    """
    tokens = []
    pad_masks = []
    while len(tokens) < batch_size:
        try:
            story = next(iterator)["text"]
        except StopIteration:
            iterator = iter(ds)
            story = next(iterator)["text"]
        ids = enc.encode(story)[:seq_len]
        real_len = len(ids)
        if real_len < seq_len:
            ids = ids + [pad_token] * (seq_len - real_len)
        tokens.append(ids)
        pad_masks.append([True] * real_len + [False] * (seq_len - real_len))
    return Tensor(tokens), Tensor(pad_masks), iterator


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    Tensor.manual_seed(args.seed)

    enc = ENC
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    iterator = iter(ds)

    model = DiffusionTransformer(
        vocab_size=VOCAB_SIZE,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )
    params = nn.state.get_parameters(model)
    optim = nn.optim.Adam(params, lr=args.lr)

    param_count = sum(p.numel() for p in params)
    print(f"Model params: {param_count / 1e6:.1f}M")
    print(
        f"Training for {args.steps} steps, batch_size={args.batch_size}, seq_len={args.seq_len}"
    )
    print(f"LR warmup: {args.warmup_steps} steps, max_grad_norm={args.max_grad_norm}")

    log_path = os.path.join(args.checkpoint_dir, "train_log.csv")
    log_file = None

    Tensor.training = True
    try:
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["step", "loss", "lr", "dt"])
        for step in range(1, args.steps + 1):
            t0 = time.monotonic()

            # LR warmup
            lr = get_lr(step, args.warmup_steps, args.lr)
            optim.lr.assign(Tensor([lr], dtype=optim.lr.dtype)).realize()

            x0, pad_mask, iterator = make_batch(
                ds, iterator, enc, args.batch_size, args.seq_len, enc.eot_token
            )
            t = Tensor.rand(args.batch_size)
            xt, mask = forward_process(x0, t, mask_token_id=VOCAB_SIZE)
            logits = model(xt, t)
            loss = compute_loss(logits, x0, mask, pad_mask)

            optim.zero_grad()
            loss.backward()
            clip_grad_norm(params, max_norm=args.max_grad_norm)
            optim.step()

            elapsed = time.monotonic() - t0
            loss_val = loss.item()
            log_writer.writerow([step, f"{loss_val:.4f}", f"{lr:.2e}", f"{elapsed:.2f}"])

            if step % args.log_every == 0:
                print(
                    f"step={step:>5d}  loss={loss_val:.4f}  lr={lr:.2e}  dt={elapsed:.2f}s"
                )
                log_file.flush()

            if step % args.save_every == 0:
                path = os.path.join(args.checkpoint_dir, f"model_{step}.safetensors")
                safe_save(get_state_dict(model), path)
                print(f"Saved checkpoint: {path}")
    finally:
        Tensor.training = False
        if log_file is not None:
            log_file.close()

    path = os.path.join(args.checkpoint_dir, f"model_{args.steps}.safetensors")
    safe_save(get_state_dict(model), path)
    print(f"Training complete. Final checkpoint: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a discrete diffusion transformer on TinyStories"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=10000)
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
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
