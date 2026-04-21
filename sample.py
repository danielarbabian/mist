import argparse
import os
import sys

from tinygrad import Tensor
from tinygrad.nn.state import load_state_dict, safe_load

from config import ENC, VOCAB_SIZE
from diffusion import sample
from model import DiffusionTransformer

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR_SCREEN = "\033[2J\033[H"


def decode_with_predictions(
    token_ids: list[int], pred_ids: list[int], mask_token_id: int
) -> str:
    """Decode tokens, showing model predictions at masked positions in dim text."""
    parts = []
    for tid, pid in zip(token_ids, pred_ids):
        if tid == mask_token_id:
            safe_pid = min(pid, VOCAB_SIZE - 1)
            parts.append(f"{DIM}{ENC.decode([safe_pid])}{RESET}")
        else:
            parts.append(f"{BOLD}{ENC.decode([tid])}{RESET}")
    return "".join(parts)


def collapse_whitespace(text: str) -> str:
    """Replace runs of whitespace (including model-predicted newlines) with single spaces,
    preserving ANSI escape codes."""
    result = []
    in_escape = False
    last_was_space = False
    for ch in text:
        if ch == "\033":
            in_escape = True
            result.append(ch)
            last_was_space = False
        elif in_escape:
            result.append(ch)
            if ch == "m":
                in_escape = False
        elif ch in (" ", "\n", "\r", "\t"):
            if not last_was_space:
                result.append(" ")
                last_was_space = True
        else:
            result.append(ch)
            last_was_space = False
    return "".join(result)


def generate(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    model = DiffusionTransformer(
        vocab_size=VOCAB_SIZE,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )
    load_state_dict(model, safe_load(args.checkpoint))

    prev_training = Tensor.training
    Tensor.training = False
    try:
        for i in range(args.num_samples):
            if args.stream:
                term_width = os.get_terminal_size().columns

                def on_step(step: int, total: int, xt: Tensor, sampled: Tensor) -> None:
                    if step % args.stream_every != 0 and step != total:
                        return
                    token_ids = xt[0].numpy().tolist()
                    pred_ids = sampled[0].numpy().tolist()
                    text = decode_with_predictions(
                        [int(t) for t in token_ids], [int(p) for p in pred_ids], VOCAB_SIZE
                    )
                    wrapped = collapse_whitespace(text)
                    n_masked = sum(1 for t in token_ids if int(t) == VOCAB_SIZE)
                    pct = 100 * (len(token_ids) - n_masked) / len(token_ids)
                    header = f" step {step}/{total}  |  {pct:.0f}% revealed "
                    sys.stdout.write(CLEAR_SCREEN)
                    sys.stdout.write(f"{DIM}{'─' * term_width}\n{header}\n{'─' * term_width}{RESET}\n\n")
                    sys.stdout.write(wrapped)
                    sys.stdout.write(f"\n\n{DIM}{'─' * term_width}{RESET}\n")
                    sys.stdout.flush()

                sample(
                    model,
                    seq_len=args.seq_len,
                    num_steps=args.num_steps,
                    vocab_size=VOCAB_SIZE,
                    temperature=args.temperature,
                    self_cond=args.self_cond,
                    on_step=on_step,
                )
                if i < args.num_samples - 1:
                    input(f"\n{DIM}Press Enter for next sample...{RESET}")
            else:
                tokens = sample(
                    model,
                    seq_len=args.seq_len,
                    num_steps=args.num_steps,
                    vocab_size=VOCAB_SIZE,
                    temperature=args.temperature,
                    self_cond=args.self_cond,
                )
                token_ids = tokens[0].numpy().tolist()
                token_ids = [min(int(t), VOCAB_SIZE - 1) for t in token_ids]
                text = ENC.decode(token_ids)
                print(f"\n--- Sample {i + 1} ---")
                print(text)
                print()
    finally:
        Tensor.training = prev_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--stream", action="store_true", help="Show denoising process in real time")
    parser.add_argument("--stream-every", type=int, default=5, help="Render every Nth step when streaming")
    parser.add_argument(
        "--self-cond",
        action="store_true",
        help="Enable self-conditioning at inference (requires model trained with --self-conditioning)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
