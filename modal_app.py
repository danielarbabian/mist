"""Modal entry point for mist training.

Run (detached, so your terminal/laptop can close):

    modal run --detach modal_app.py --steps 20000 --run-name v2

Training runs on Modal's infrastructure. Checkpoints and the train log are
written to the `mist-checkpoints` volume under /<run_name>/. Pull them with:

    modal volume get mist-checkpoints v2/model_20000.safetensors ./
    modal volume get mist-checkpoints v2/train_log.csv ./

For a completion ping, pass --slack-webhook-url=$SLACK_WEBHOOK_URL.
"""

import os

import modal

app = modal.App("mist")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.13"
    )
    .pip_install(
        "tinygrad",
        "tiktoken",
        "datasets",
        "numpy",
        "requests",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("model", "diffusion", "train", "config")
)

checkpoints_vol = modal.Volume.from_name("mist-checkpoints", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("mist-hf-cache", create_if_missing=True)

CHECKPOINT_PATH = "/checkpoints"
HF_CACHE_PATH = "/root/.cache/huggingface"


def _notify(webhook_url: str | None, message: str) -> None:
    if not webhook_url:
        return
    try:
        import requests

        requests.post(webhook_url, json={"text": message}, timeout=10)
    except Exception as exc:
        print(f"webhook post failed: {exc}")


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 60 * 8,
    volumes={
        CHECKPOINT_PATH: checkpoints_vol,
        HF_CACHE_PATH: hf_cache_vol,
    },
)
def train_on_modal(
    steps: int,
    batch_size: int,
    lr: float,
    min_lr: float,
    lr_schedule: str,
    ema_decay: float,
    seq_len: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    seed: int,
    warmup_steps: int,
    max_grad_norm: float,
    log_every: int,
    save_every: int,
    val_every: int,
    val_size: int,
    run_name: str,
    slack_webhook_url: str | None,
) -> None:
    import argparse

    from train import train

    run_dir = os.path.join(CHECKPOINT_PATH, run_name)

    args = argparse.Namespace(
        batch_size=batch_size,
        lr=lr,
        min_lr=min_lr,
        lr_schedule=lr_schedule,
        ema_decay=ema_decay,
        steps=steps,
        seq_len=seq_len,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        checkpoint_dir=run_dir,
        log_every=log_every,
        save_every=save_every,
        val_every=val_every,
        val_size=val_size,
        seed=seed,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
    )

    try:
        train(args)
        _notify(
            slack_webhook_url, f"mist: `{run_name}` finished ({steps} steps)"
        )
    except Exception as exc:
        _notify(slack_webhook_url, f"mist: `{run_name}` FAILED — {exc}")
        raise
    finally:
        checkpoints_vol.commit()
        hf_cache_vol.commit()


@app.local_entrypoint()
def main(
    steps: int = 20000,
    batch_size: int = 64,
    lr: float = 3e-4,
    min_lr: float = 0.0,
    lr_schedule: str = "cosine",
    ema_decay: float = 0.999,
    seq_len: int = 128,
    dim: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    seed: int = 42,
    warmup_steps: int = 500,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
    save_every: int = 5000,
    val_every: int = 500,
    val_size: int = 256,
    run_name: str = "default",
    slack_webhook_url: str | None = None,
) -> None:
    train_on_modal.remote(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        min_lr=min_lr,
        lr_schedule=lr_schedule,
        ema_decay=ema_decay,
        seq_len=seq_len,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        seed=seed,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        log_every=log_every,
        save_every=save_every,
        val_every=val_every,
        val_size=val_size,
        run_name=run_name,
        slack_webhook_url=slack_webhook_url,
    )
