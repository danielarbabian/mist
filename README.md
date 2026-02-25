# mist

A tiny diffusion language model built from scratch in [tinygrad](https://github.com/tinygrad/tinygrad).

Text doesn't have to be generated left-to-right. `mist` starts from pure noise (every token masked) and iteratively denoises until coherent text emerges.

Based on [MDLM](https://arxiv.org/abs/2406.07524) (Masked Diffusion Language Models).

## Why

Autoregressive generation (GPT-style, predict next token) is one way to do language modeling. Diffusion is another method where you start from noise and refine toward signal. Image diffusion models proved this works incredibly well for continuous data. Discrete diffusion applies the same idea to token sequences.

This project exists to build that from first principles in a minimal framework and understand every piece. Not trying to beat any benchmarks, just trying to really get how this works for my own learning.

## How It Works

During training, we take clean text, sample a random timestep, and mask tokens at a rate determined by that timestep. The model's job is to learn to predict what's behind the masks.

```
"the cat sat on the mat"    →    "the [M] sat [M] the [M]"    →    "[M] [M] [M] [M] [M] [M]"
         clean                        partial noise                      full noise
```

At inference time, we start from a fully masked sequence and iteratively unmask it. Each step, the model predicts all positions at once and we reveal the ones it's most confident about. Over ~10-20 steps, text emerges from the mist.

```
"[M] [M] [M] [M] [M] [M]"  →  "[M] cat [M] [M] [M] [M]"  →  "the cat sat on the mat"
```

Because all positions are predicted in parallel, there's no left-to-right bottleneck like you'd have with autoregressive models.

## Architecture

The core is a small transformer with timestep conditioning. The forward and reverse diffusion processes are just math, so the denoising model is the only thing actually being trained.

| Component          | What it does                                                             |
| ------------------ | ------------------------------------------------------------------------ |
| Token Embedding    | Maps vocab to vectors, with a learned `[MASK]` embedding                 |
| Timestep Embedding | Sinusoidal encoding fed through an MLP, injected via adaptive layer norm |
| Transformer Blocks | Self-attention + FFN + LayerNorm, nothing exotic                         |
| Output Head        | Projects to vocab logits, with CE loss computed on masked positions only |

Training dataset TBD. Probably [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) or a WikiText subset, just needs to be small enough to train on a single GPU :()

## Plan

### Phase 1: Core

- [ ] Transformer with timestep conditioning
- [ ] Absorbing-state forward process (masking schedule)
- [ ] Reverse process (iterative unmasking)
- [ ] Training loop
- [ ] Sampling / generation

### Phase 2: Begin training

- [ ] Train on TinyStories or similar
- [ ] Tune noise schedule, sampling steps
- [ ] Get coherent generations

### Phase 3: Make It Cool

- [ ] Step-by-step denoising visualization
- [ ] Loss curves
- [ ] Comparison of generation quality vs number of denoising steps

## Project Structure

```
mist/
├── model.py       # transformer + timestep conditioning
├── diffusion.py   # noise schedule, forward corruption, reverse sampling
├── train.py       # training loop
└── sample.py      # generate text, visualize the denoising process
```

## Key Decisions

**Why tinygrad?** The whole point is to implement everything myself. Tinygrad's op set is minimal enough that you can't hide behind abstractions. If it works, you understand it and I've been wanting to try tinygrad for a while.

**Why MDLM specifically?** Absorbing-state diffusion (tokens get masked, model unmasks them) is the most intuitive discrete diffusion formulation. It's conceptually close to BERT-style masked language modeling, but with a proper diffusion framework top. It's also way easier to debug than score-based approaches like SEDD.

**Why not autoregressive?** Fewer people have built a diffusion LM from scratch, and even fewer in tinygrad. Diffusion generation also has some genuinely interesting properties like parallel decoding, the ability to revise all positions simultaneously, and controllable generation via guidance.

## References

- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) (Sahoo et al., 2024)
- [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) (Austin et al., 2021) - D3PM, foundational discrete diffusion work
- [Score Entropy Discrete Diffusion](https://arxiv.org/abs/2310.16834) (Lou et al., 2024) - SEDD, alternative approach

## License

MIT
