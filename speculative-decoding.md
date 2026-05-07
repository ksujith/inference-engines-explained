# Speculative decoding

A reading guide. Companion to the main [README](./README.md) — the README
explains how a serving engine works; this page explains the single biggest
inference-acceleration trick layered on top.

> Truthful note: mini-sglang's baseline does **not** implement speculative
> decoding. SGLang upstream, vLLM, and TensorRT-LLM do. We'll explain how it
> works and where it would slot into the §1–12 architecture.

---

## 1. Why decode is slow in the first place

A decode step reads **every weight in the model** out of HBM to produce
**one token**. The actual arithmetic with those weights is tiny. Your fancy
GPU sits at single-digit utilization — the bottleneck is "how fast can I
stream 70B parameters from memory", not "how fast can I multiply".

The crucial observation that makes speculative decoding possible:

> A forward pass on **K tokens** costs almost the same wall-clock as on **1
> token**. You read the weights once either way.

Decode wastes that. It produces 1 token per weight-stream. Speculative
decoding fills the wasted compute with useful work.

---

## 2. The trick, in one picture

A cheap **draft** model proposes K candidate tokens. The big **target**
model runs **once** on the K-token sequence — in that single forward pass it
produces its own prediction at every position. Compare draft vs target,
left-to-right. Keep the matching prefix; on the first mismatch, take the
target's token and throw the rest away.

```
              draft (small, fast, autoregressive)
                              │
 "United States of ___"  ──►  ["America", ".",  " It",  " is"]
                                                                │
                                                                ▼
                          target — one forward pass on all 4 + bonus position
                                                                │
                                                                ▼
                          target says:  ["America", ".",  " The", ...   ]
                                              ✓         ✓        ✗
                                              └────────┴── accept ──┘
                                                                  │
                                            continue from target's "The"
```

In this step we got 2 drafted tokens **plus** 1 free token from the target
(its prediction at the rejection point), in 1 target forward + 4 draft
forwards. Net win whenever:

```
K · draft_cost  +  target_cost   <   (k_accepted + 1) · target_cost
```

A lot of next tokens are obvious from local context — "United States of ___",
boilerplate code, repeated phrasing. The draft nails these; the target only
has to actually think at the genuinely uncertain branches.

---

## 3. Why it doesn't degrade quality

The surprising part. With the right accept/reject rule (Leviathan et al.,
2023), the output distribution is **provably identical** to sampling from the
target alone.

The rule is standard rejection sampling. For draft probability `q(x)` and
target probability `p(x)` at a given position:

- If `p(x) ≥ q(x)`: **accept**.
- Else: accept with probability `p(x) / q(x)`. On reject, **resample** from
  the corrected distribution `normalize( max(0, p − q) )`.

The math guarantees the marginal distribution at every accepted position
equals `p`. Not approximate, not "close enough" — exact. This is why people
react with "wait, this is too good to be true."

---

## 4. The core loop, ~30 lines

```python
def speculative_step(prefix, draft, target, K):
    # 1. Draft proposes K tokens, autoregressively, cheaply.
    drafted, q_probs = [], []
    seq = list(prefix)
    for _ in range(K):
        q = draft(seq)              # distribution over vocab
        x = sample(q)
        drafted.append(x); q_probs.append(q)
        seq.append(x)

    # 2. Target verifies all K in ONE forward pass.
    #    Returns K+1 distributions: K for verification, +1 bonus.
    p_probs = target(prefix + drafted)

    # 3. Accept / reject, left to right.
    out = []
    for i, x in enumerate(drafted):
        if random() < min(1.0, p_probs[i][x] / q_probs[i][x]):
            out.append(x)                                    # accept
        else:
            corrected = normalize(relu(p_probs[i] - q_probs[i]))
            out.append(sample(corrected))                    # reject + resample
            return out                                       # stop early

    # 4. All K accepted → free bonus token from target's last distribution.
    out.append(sample(p_probs[K]))
    return out
```

Per step you produce between **1 and K+1** tokens. The expected speedup
(Leviathan 2023) is:

```
            1 − α^(K+1)
speedup ≈ ─────────────────
          (1 − α)(1 + K·c)
```

where:
- `α` = acceptance rate (probability the target accepts a drafted token)
- `c` = `draft_cost / target_cost`

Plug in α=0.7, K=4, c=0.05 → **~2.3×**. Plug in α=0.4 → barely above 1×. Below
some threshold it's actually **slower** than vanilla decoding. Always
benchmark; don't take the speedup on faith.

---

## 5. The one knob that matters: acceptance rate `α`

Everything in production speculative decoding is a fight over α.

| Lever                  | Effect                                                        |
| ---------------------- | ------------------------------------------------------------- |
| Domain match           | Draft trained on similar data → α up. Code draft for code.    |
| Draft size             | Bigger draft → α up, but `c` up. Sweet spot ~1/30–1/10 target.|
| Temperature            | Higher T flattens distributions → more disagreement → α down. |
| K (draft length)       | Bigger K hits diminishing returns; each extra pos. drops α.   |

Practical α numbers in published systems land in the 0.5–0.8 range depending
on workload. Greedy / low-temperature inference benefits the most.

---

## 6. Variants

The vanilla version uses a separate small LM. The interesting work since
2023 has been making the *draft* cheaper or training-free.

| Variant          | Draft mechanism                                          | Trains target?    |
| ---------------- | -------------------------------------------------------- | ----------------- |
| Vanilla (2023)   | Separate small LM (e.g. 1B drafts 70B)                   | No                |
| Medusa           | Extra prediction heads bolted onto frozen target         | Heads only        |
| EAGLE / EAGLE-2  | Autoregressive head conditioned on target's hidden state | Heads only        |
| MTP              | Target *trained* to predict `t+1, t+2, …` jointly        | Yes — from scratch|

**MTP** is the production endgame. DeepSeek-V3 uses it. The big advantage:
no separate draft model in HBM, draft cost collapses to ~free. The catch:
you have to retrain the target to support it — you can't bolt MTP onto an
existing checkpoint.

---

## 7. Where it would slot into the engine (§1–12 of the main README)

| Layer                          | Change                                                                                  |
| ------------------------------ | --------------------------------------------------------------------------------------- |
| `engine.py` decode forward     | Input `[B, 1]` → `[B, K]`. Same kernel, longer sequence.                                |
| `kvcache/`                     | Page alloc reserves K slots per request. **Roll back** unused slots on partial reject.  |
| `scheduler/decode.py`          | A decode step now advances `cached_len` by `k_accepted` (1..K+1), not always 1.         |
| `scheduler/scheduler.py`       | Structural shape unchanged. Continuous batching still works.                            |

Two things deserve a warning:

1. **KV rollback is the hardest part of a real implementation**, harder than
   the accept/reject math. You allocated K pages worth of KV; if only 2 of K
   tokens stuck, the other K−2 worth of KV must be released and the page
   table walked back. Get this wrong and you corrupt subsequent decode steps.
2. **Batching speculative requests with non-speculative ones is awkward** —
   they have different per-step token counts, so the engine either separates
   the batches or pads everything to the spec batch's K.

---

## 8. Try it yourself

Implement vanilla speculative decoding from scratch. **No coding agent for
the core loop** — write it yourself, top to bottom. Pick a target/draft pair
sharing a tokenizer (e.g. Llama-3.2-1B drafting Llama-3.1-8B).

Goals:

1. The 30-line accept/reject loop from §4. Get the rejection-sampling math
   right — verify empirically that the output distribution matches greedy /
   pure-target sampling on a fixed seed.
2. A throughput script: tokens/sec vs. plain decode on the same prompts.
   Measure under different temperatures.
3. Plot acceptance rate by position within the K-window. You'll see it decay
   — that's the diminishing-returns curve from §5.

You'll learn more in an afternoon than from any paper figure. Then go read
MTP — once you've felt the rollback pain, the appeal of joint-trained heads
is obvious.

---

## References

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, 2023 — [arxiv:2211.17192](https://arxiv.org/abs/2211.17192). The original.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023 — [arxiv:2302.01318](https://arxiv.org/abs/2302.01318). DeepMind's parallel discovery.
- Cai et al., *Medusa*, 2024 — [arxiv:2401.10774](https://arxiv.org/abs/2401.10774).
- Li et al., *EAGLE*, 2024 — [arxiv:2401.15077](https://arxiv.org/abs/2401.15077).
- DeepSeek-V3 technical report — MTP at production scale.
