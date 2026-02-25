# Building a Diffusion Language Model from Scratch

> _This is a walkthrough of how I built a discrete diffusion language model in tinygrad. I am by no means an expert, rather I find myself learning best when I get involved hands-on. I am learning this as I go, and writing it down so that you can learn it too. If something feels hand-wavy, it probably is, and I will try to be honest about the parts I am still figuring out._

---

## Why Diffusion for Text?

If you have spent any time in the deep learning world you have almost certainly encountered autoregressive language models, the GPT family being the most famous example. These models generate text one token at a time, always moving left to right, conditioning each new token on everything that came before it. This approach is elegant and it works extraordinarily well, which is exactly why almost everyone uses it.

But if you think about it for a moment, this is a somewhat strange constraint. When you write an essay you do not produce it strictly left to right, finalising each word before moving on to the next. You sketch out ideas, revise, move things around, and gradually refine the whole piece until it is coherent.

**Diffusion models work much more like that.** They start from noise and progressively refine the entire output in parallel, revisiting every position at every step.

Diffusion has already proven itself spectacularly in the continuous domain. Models like Stable Diffusion and DALL-E generate images by starting from pure Gaussian noise and iteratively denoising it into a coherent picture. The question that motivated this project was whether you could do the same thing for text, which is a fundamentally discrete object. You cannot add Gaussian noise to a sentence the same way you can to a pixel grid, so the approach needs to be rethought from the ground up.

It turns out that you can, and the specific formulation I chose to implement is called **MDLM** (Masked Diffusion Language Models), from Sahoo et al. (2024). The reason I picked MDLM over other discrete diffusion approaches is that it is conceptually the simplest: the "noise" is just masking tokens out, and the model learns to put them back. If you have ever seen BERT's masked language modelling objective, the core idea will feel very familiar, though the diffusion framework around it is what makes generation possible.

[Mercury 2's release](https://x.com/_inception_ai/status/2026297527843409933) inspired me heavily to make this project, although I've been learning about it for slightly longer.

---

## The Core Idea

The entire framework rests on two processes that mirror each other.

### Forward Process (Destruction)

Takes a clean sentence and gradually destroys it by replacing tokens with a special `[MASK]` token.

```
t=0.0  "the cat sat on the mat"         ← fully clean
t=0.3  "the cat [M] on the [M]"         ← some tokens masked
t=0.7  "[M] [M] [M] on [M] [M]"         ← most tokens masked
t=1.0  "[M] [M] [M] [M] [M] [M]"         ← fully destroyed
```

### Reverse Process (Generation)

Starts from pure noise and iteratively unmasks tokens until coherent text emerges.

```
step 10  "[M] [M] [M] [M] [M] [M]"      ← pure noise
step 7   "[M] cat [M] [M] [M] [M]"      ← first tokens appear
step 3   "the cat [M] on the [M]"       ← taking shape
step 0   "the cat sat on the mat"       ← done
```

### What makes this a diffusion model?

What separates this from just "mask and predict" is the **noise schedule** and the probabilistic framework connecting the two processes. During training we do not just mask at a fixed rate; we sample a random timestep and mask at whatever rate corresponds to that timestep on a carefully chosen schedule. This forces the model to learn to denoise from every possible corruption level, from nearly clean inputs to nearly fully masked ones.

---

## The Noise Schedule

The noise schedule is a function $\alpha(t)$ that tells us the probability that any given token has survived (has _not_ been masked) at timestep $t$. It needs to satisfy two boundary conditions:

$$
\alpha(0) = 1 \quad \text{(all tokens intact)}
$$

$$
\qquad \alpha(1) = 0 \quad \text{(everything masked)}
$$

Between these endpoints, $\alpha$ should decrease monotonically. The simplest option would be linear, $\alpha(t) = 1 - t$, but in practice the **cosine schedule** works better:

$$
\alpha(t) = \cos^2\!\left(\frac{\pi t}{2}\right)
$$

### Why cosine beats linear

The cosine schedule spends more of its range in the middle corruption levels, where the learning signal is richest. A linear schedule moves at a constant rate, which means the model spends equal training time on very easy reconstructions (barely anything masked) and very hard ones (almost everything masked). The cosine curve changes slowly near $t=0$ and $t=1$ but moves quickly through the middle range, effectively concentrating training effort on the corruption levels that are neither trivially easy nor impossibly hard. I still struggle to understand a lot of this math but it somewhat makes sense to me the more I work on this.

In code this ends up being a single line, which is one of those satisfying moments where the math collapses into almost nothing:

```python
def noise_schedule(t):
    return (t * math.pi / 2).cos() ** 2
```

---

## The Forward Process

Given a clean sequence of tokens $x_0$ and a timestep $t$, the forward process produces a corrupted version $x_t$ by independently masking each token with probability $1 - \alpha(t)$. "Independently" is the key word here: each token's fate is decided by its own coin flip, regardless of what happens to its neighbours.

The procedure is:

1. Compute $\alpha(t)$ from the noise schedule
2. For each token position, draw a uniform random number $u \sim \text{Uniform}(0, 1)$
3. If $u > \alpha(t)$, replace that token with `[MASK]`, otherwise keep the original

```python
def forward_process(x0, t, mask_token_id):
    alpha = noise_schedule(t).unsqueeze(1)        # (B, 1)
    mask = Tensor.rand(*x0.shape) > alpha          # True where masked
    xt = Tensor.where(mask, mask_token_id, x0)
    return xt, mask
```

One thing that took me a moment to internalise is that this independence assumption means the forward process has no notion of linguistic structure whatsoever. It does not care that masking both the subject and verb of a sentence makes reconstruction much harder than masking two adjectives. The model has to learn to handle whatever random pattern of masking the forward process throws at it, which is precisely what gives it the generality needed for generation.

> **Why return the mask?** The mask tells the training loop which positions were corrupted. The loss function should only penalise predictions at masked positions, because there is no useful signal from asking the model to predict tokens that were handed to it in the clear.

---

## The Model

The denoising model is the only component with learnable parameters. It is a fairly standard transformer with one addition: **timestep conditioning**. It takes a partially masked token sequence $x_t$ and a scalar timestep $t$, and outputs logits over the vocabulary for every position.

### Why the model needs to know the timestep

Consider what happens if you do not tell the model what timestep it is at. When 90% of tokens are masked ($t$ close to 1), the model should hedge its bets and produce relatively uniform predictions, because there is very little information to work with. When only 10% of tokens are masked ($t$ close to 0), the model should be highly confident, because the surrounding context makes the answer nearly unambiguous.

Without knowing $t$, the model has no way to calibrate the sharpness of its predictions, and you end up with something that is either perpetually overconfident or perpetually uncertain.

### Timestep embedding

The timestep is a scalar in $[0, 1]$, and we need to turn it into a rich vector that the transformer can condition on. We use **sinusoidal embeddings**, the same technique originally introduced in "Attention Is All You Need" for encoding positions:

$$
\text{emb}(t)_{2i} = \sin(t \cdot \omega_i)
$$

$$
\qquad \text{emb}(t)_{2i+1} = \cos(t \cdot \omega_i)
$$

where $\omega_i = e^{-\log(10000) \cdot i / d}$ are geometrically spaced frequencies. The 10,000 is not as arbitrary as it looks, what it controls is the range of wavelengths in the embedding: the lowest frequency dimension oscillates very slowly (wavelength of $2\pi \times 10000$), while the highest frequency dimension oscillates rapidly (wavelength of $2\pi$). By spacing them geometrically across this range, you get a set of sinusoids that can distinguish between values of $t$ at both coarse and fine granularity. A smaller base (say 100) would compress all the wavelengths into a narrow band and waste capacity on redundant frequencies, while a much larger base would spread them so thin that nearby values of $t$ become hard to tell apart. The original "Attention Is All You Need" paper landed on 10,000 because it worked well for position encodings over typical sequence lengths, and it turns out to transfer nicely to timestep encodings too.

After the sinusoidal embedding, we pass the result through a small two-layer MLP with a GELU activation, which gives the model a chance to learn its own internal representation of the timestep rather than being stuck with the raw sinusoidal features.

### Adaptive Layer Normalization (AdaLN)

The timestep conditioning is injected into the transformer via **Adaptive Layer Normalization**. In standard layer norm you normalise the hidden states and apply a learned scale and shift. In AdaLN, the scale and shift are _predicted from the timestep embedding_:

$$
\text{AdaLN}(x, t) = (1 + \gamma(t)) \cdot \text{LayerNorm}(x) + \beta(t)
$$

where $\gamma(t)$ and $\beta(t)$ come from a linear projection of the timestep embedding. This is the same conditioning mechanism used in DiT (Diffusion Transformers for images), and it works well because it lets the timestep modulate every layer of the network without adding much complexity.

### Architecture summary

| Component                     | What it does                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| Token + Position Embedding    | Maps input tokens and positions to vectors, with a learned `[MASK]` embedding         |
| Sinusoidal Timestep Embedding | Encodes scalar $t$ into a vector, refined through a 2-layer MLP                       |
| N x Transformer Block         | Self-attention + FFN + AdaLN conditioning, residual connections                       |
| Output Head                   | Projects to vocab logits (no mask token in output, we never want to predict `[MASK]`) |

The rest of the transformer is completely standard: multi-head self-attention, feed-forward networks with GELU activations, and residual connections. There is nothing overly complex here and that is intentional. The whole point was to see how far you can get with the simplest possible architecture, where all the interesting behaviour comes from the diffusion framework rather than architectural tricks.

---

## The Reverse Process

This is the process that results in the actual generation.

1. Start with a fully masked sequence: every position is `[MASK]`
2. For each step from $t = 1$ down to $t = 0$:
   - Run the model on the current sequence at timestep $t$ to get logits
   - Convert logits to token predictions (argmax or sampling)
   - Decide which masked positions to unmask at this step
   - Replace those positions with the predicted tokens
3. At the final step, unmask everything that remains

### How many tokens to unmask at each step?

This is where the noise schedule reappears. If we are going from timestep $t$ to $t_\text{prev}$, the probability that any currently masked position gets unmasked is:

$$
p_\text{unmask} = \frac{\alpha(t_\text{prev}) - \alpha(t)}{1 - \alpha(t)}
$$

The denominator $1 - \alpha(t)$ accounts for the fact that we are only considering positions that are _currently masked_. The numerator $\alpha(t_\text{prev}) - \alpha(t)$ is the "budget" of new unmaskings needed at this step to stay on schedule.

This formula ensures that running the full reverse process from $t=1$ to $t=0$ results in roughly the right number of tokens unmasked at every intermediate point, matching what the model was trained to expect.

```python
alpha_t = noise_schedule(t)
alpha_prev = noise_schedule(t_prev)
unmask_prob = (alpha_prev - alpha_t) / (1 - alpha_t)
```

### Variable step counts

One thing I find beautiful about this setup is that the number of denoising steps is a **free parameter at inference time**. You can use 10 steps for fast generation or 100 steps for more careful refinement, and the formula above automatically adjusts how many tokens get revealed at each step. More steps means smaller increments, which generally produces higher quality output because the model gets more opportunities to condition on previously revealed tokens.

---

## Training

The training loop ties everything together and is surprisingly simple:

1. Sample a batch of clean text sequences $x_0$
2. Sample random timesteps $t \sim \text{Uniform}(0, 1)$, one per sequence
3. Run the forward process to get corrupted sequences $x_t$ and masks
4. Feed $x_t$ and $t$ through the model to get logits
5. Compute cross-entropy loss **on masked positions only**
6. Backpropagate and update

### Why loss on masked positions only?

There is no useful gradient signal from unmasked positions. If a token was not masked, the model can just copy it from the input embedding. Predicting visible tokens correctly tells us nothing about the model's ability to denoise; it is the masked positions where the model has to actually reason about what belongs there, and those are the only positions that should contribute to the loss.

### Why uniform timestep sampling?

Each training step, the model sees a different corruption level, and over the course of training it learns to handle all of them. This is fundamentally different from training a single BERT-style model with a fixed 15% masking rate. The diffusion training procedure creates a model that can denoise from _any_ corruption level, which is what the iterative generation process requires.

---

## Putting It All Together

What strikes me most about this approach is how much of the heavy lifting is done by the framework rather than the model. The transformer itself is completely generic; there is nothing in its architecture that is specific to diffusion. All the diffusion-specific behaviour comes from _how_ you train it (random timesteps, masked loss) and _how_ you sample from it (iterative unmasking). The model just learns to predict masked tokens given a corruption level, and the diffusion framework turns that simple capability into a full generative model.

If you are used to autoregressive models, the parallel nature of diffusion generation takes some getting used to. There is no sequential dependency between positions, which means the model can revise its prediction for any position at any step based on what has been revealed elsewhere. Whether this actually produces better text than autoregressive generation is still an open research question, and for a small model trained on limited data the answer is probably no. But the mechanism itself is interesting, and I think building it from scratch is the best way to develop an intuition for how and why it works.

---

## References

1. Sahoo, S., Arriola, M., Schiff, Y., et al. (2024). [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524). _arXiv:2406.07524_.
2. Austin, J., Johnson, D., Ho, J., et al. (2021). [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006). _arXiv:2107.03006_.
3. Peebles, W. & Xie, S. (2023). [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). _ICCV 2023_.
4. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). _NeurIPS 2017_.
