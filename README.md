# Inference engines, taught through mini-sglang

A reading guide that maps the concepts of modern LLM inference engines onto the
[`sgl-project/mini-sglang`](https://github.com/sgl-project/mini-sglang) codebase
— small enough to read end-to-end, complete enough to run real LLMs.

> File and line references throughout assume the layout of mini-sglang at the
> time of writing. If the upstream repo has moved on, the *concepts* still map;
> the line numbers may not.

---

## 1. What an inference engine actually does

A user sends a prompt. The engine must:

1. Tokenize → run the model on the prompt → produce one token at a time → detokenize → stream back.
2. Do this for **many concurrent users**, sharing a single GPU's compute and memory.
3. Hide every source of latency it can: kernel launches, Python overhead, recomputation, communication.

Almost every design decision in mini-sglang is in service of one of those three goals. Keep that lens on as we go.

---

## 2. The process layout (`docs/structures.md`, `server/`)

The engine is **multi-process**, not multi-thread, because Python's GIL and CUDA contexts don't play well together:

```
User → API Server (FastAPI)
         │  zmq
         ▼
       Tokenizer worker
         │  zmq
         ▼
       Scheduler rank 0  ◄──► Scheduler rank 1, 2, ... (one per GPU; NCCL between them)
         │ owns →  Engine (model + KV cache + attention backend)
         │  zmq
         ▼
       Detokenizer → API Server → User (streamed)
```

- **Control plane** (small messages: "here's a request", "here's a token") goes through ZMQ.
- **Data plane** (big tensors between GPUs for tensor parallelism) goes through NCCL.

Why split tokenizer/detokenizer out? CPU-bound work that would otherwise stall the GPU loop.

Files: `server/launch.py` spawns everything; `server/api_server.py` is the FastAPI front; `tokenizer/`; `scheduler/scheduler.py`; `engine/engine.py`.

---

## 3. The two phases: **prefill** vs **decode**

This is THE distinction in LLM inference. Internalize it:

|                           | Prefill                                | Decode                              |
| ------------------------- | -------------------------------------- | ----------------------------------- |
| Input                     | the whole prompt (e.g. 2,000 tokens)   | exactly 1 token (the previous one)  |
| Compute per request       | proportional to prompt length          | constant                            |
| Bottleneck                | compute (GEMM-bound)                   | memory bandwidth (KV cache reads)   |
| Per-token cost            | cheap (lots of parallelism)            | expensive (no parallelism)          |

Same model weights, same forward pass — but the *shape* of the work is so
different that the engine literally batches them separately. See `core.py`
(`Batch.phase` is `"prefill"` or `"decode"`) and the two managers
`scheduler/prefill.py` and `scheduler/decode.py`.

The whole point of batching decode requests together is that since each request
only contributes 1 token, you get GEMM efficiency only by stacking many
requests' single tokens into one matrix multiply.

---

## 4. The KV cache — the central abstraction

When a transformer processes token `t`, it computes Key and Value vectors at
every layer. **Future tokens need those K and V** to attend to past tokens. So
you save them. That saved tensor is the KV cache, and it dominates memory and
bandwidth in serving.

Mini-sglang manages it as **paged memory**, like a virtual-memory system:

- A "page" holds K+V for `page_size` consecutive tokens of one request, at all
  layers (`engine/engine.py` calculates `cache_per_page`).
- A **page table** maps `(request_idx, position)` → physical KV cache slot
  (`engine/engine.py`, `core.py`).
- Fragmentation is solved exactly the way OS virtual memory solves it: any free
  page can serve any request.

Why pages and not flat per-request buffers? Because requests have unpredictable
lengths, finish at unpredictable times, and you want to fit as many as possible.
Pages let you start a new request the moment any pages free up. (Original idea:
vLLM's PagedAttention.)

Files: `kvcache/mha_pool.py` (the page pool itself), `kvcache/radix_cache.py`
(the smart cache manager — see §7), `kvcache/naive_cache.py` (the dumb one, for
comparison).

---

## 5. The request lifecycle and the scheduler loop

Each request is a `Req` (`core.py`). Critical fields:

- `input_ids` — all tokens seen so far (prompt + generated)
- `cached_len` — how many of those are already in the KV cache
- `device_len` — how many are on the GPU's page table
- `output_len` — how many more we still want to generate

Three invariants: `cached_len ≤ device_len ≤ max_device_len`. Every state
transition (prefill chunk done, one decode step done) is just bumping these
numbers.

The scheduler loop is `scheduler/scheduler.py:normal_loop` (and `overlap_loop`
for the optimized version):

```python
def normal_loop(self):
    receive new user messages from zmq        # control plane
    forward_input = schedule_next_batch()     # decide what to run
    if forward_input:
        ongoing = self._forward(forward_input)  # GPU forward
    self._process_last_data(ongoing)          # sample, append, free, reply
```

`_schedule_next_batch` is the policy: prefill-first, decode otherwise. This is
**continuous batching** — at every step you reconsider the batch composition.
New arrivals join, finished requests leave, no batch waits for a slow member.

---

## 6. Chunked prefill (`scheduler/prefill.py`)

A 32k-token prompt would dominate compute, blocking decode for everyone.
Solution: split prefill into chunks bounded by `prefill_budget`. The first
chunks become a `ChunkedReq` — same as a `Req` except it never gets sampled,
never produces output, and never enters the decode set. Only when the last
chunk fills does it become a real `Req` ready for decode.

`PrefillAdder.try_add_one` is where the budget gets spent, page allocation gets
attempted, and the request either makes it into the batch or waits. This is
also where you see the budget split between *new* prefill tokens and *space
reserved* for the decode tokens already in flight (`reserved_size` arg).

Why chunked prefill matters: it converts a latency disaster ("one big user
blocks everyone") into smooth interleaving with decode batches.
Reference: [Sarathi-Serve](https://arxiv.org/abs/2403.02310).

---

## 7. Radix cache: the prefix-sharing trick (`kvcache/radix_cache.py`)

If two requests share a prompt prefix (system prompt, few-shot examples, RAG
context), the KV cache for that prefix is identical. Computing it twice is pure
waste.

Solution: store cached KV in a **radix tree** keyed by token IDs. When a new
request arrives, walk the tree to find the longest matching prefix
(`get_match_len`). That match becomes its `cached_len`, so prefill only computes
the *novel* suffix. The tree splits on divergence (`split_at`), reference-counts
shared nodes, and evicts cold leaves under memory pressure (LRU-ish via
timestamps and a heap).

This is `prefill.py`'s `match_req` call — every new request goes through it
before any compute is scheduled.

---

## 8. Sampling and the per-step "complete" (`engine/engine.py`)

After `model.forward()`:

```python
logits = ...                    # [batch_size, vocab_size]
next_tokens_gpu = sampler.sample(logits)
next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
copy_done_event.record(stream)
```

The non-blocking D→H copy + CUDA event is important: the scheduler can keep
launching the next batch on the GPU while the previous batch's tokens are still
being copied to CPU for EOS-checking. That's the seed of overlap scheduling.

Then for each request: `req.complete_one()` → cached_len catches up, device_len
advances by 1.

---

## 9. CUDA graphs (`engine/graph.py`, `engine/engine.py`)

In decode, batches are tiny (often just 1 token per request). Python and
CUDA-launch overhead can be **larger than the compute itself**. CUDA graphs
solve this: capture the full forward pass once, then "replay" it with new input
pointers. One launch instead of thousands.

Mini-sglang captures graphs at a discrete set of batch sizes (`cuda_graph_bs`),
then *pads* incoming batches up to the nearest captured size using the
`dummy_req`. This is why you'll see `padded_reqs` everywhere.

`engine.py` decides per-step: replay the graph if the batch is graph-shaped,
else fall back to eager.

---

## 10. Overlap scheduling (`scheduler/scheduler.py:overlap_loop`)

Look at this carefully:

```python
def overlap_loop(self, last_data):
    receive_msg(...)
    forward_input = schedule_next_batch()      # CPU work
    if forward_input:
        with self.engine_stream_ctx:
            ongoing = self._forward(...)       # GPU work, on engine stream
    self._process_last_data(last_data)         # CPU work for the PREVIOUS batch
    return ongoing
```

The function returns *this* batch's pending result. Next iteration, we schedule
a *new* batch on the GPU **while** processing the previous batch's tokens on the
CPU. Two CUDA streams and a CUDA event make this safe.

Net effect: CPU latency (token decoding, request bookkeeping, message passing)
is fully hidden behind GPU compute. This is what gets you from ~70% GPU
utilization to >95%.
Reference: [NanoFlow](https://arxiv.org/abs/2408.12757).

---

## 11. Tensor parallelism (briefly)

`distributed/` and the TP-aware layers in `layers/linear.py`. The trick: split
each big matrix multiply across GPUs along the inner or outer dim, then
`all_reduce` or `all_gather` to stitch results. Each GPU has its own
scheduler+engine; rank 0 owns user I/O; ranks 1..N-1 mirror its scheduling
decisions over a CPU group.

KV cache is also sharded — each rank only holds its own attention heads' KV
(`engine.py` divides `num_kv_heads` by `tp_info.size`).

---

## A reading order if you want to internalize this

1. `docs/structures.md` and `docs/features.md` — 5 minutes, big picture.
2. `core.py` (~140 lines) — the `Req`/`Batch`/`Context` data model. Everything else moves these around.
3. `scheduler/scheduler.py` — `normal_loop` first, then `overlap_loop`. Trace one request through.
4. `scheduler/prefill.py` and `scheduler/decode.py` — the actual scheduling policy.
5. `engine/engine.py` — model init, KV-cache sizing, `forward_batch`.
6. `kvcache/radix_cache.py` — the prefix-sharing data structure.
7. `attention/fa.py` or `attention/fi.py` — how page tables become attention kernel arguments.
8. `engine/graph.py` — CUDA graph capture/replay.
9. `models/llama.py` — see how a real model is plugged into this skeleton.

---

## Suggested exercises

- **Trace one request end-to-end.** Start at `scheduler.py`'s `_process_one_msg`
  for a `UserMsg` and follow it through scheduling, prefill, decode, until a
  `DetokenizeMsg` goes out.
- **Understand the page table.** Read `engine.py` page-table init, then
  `kvcache/mha_pool.py`, then how `_make_input_tuple` and `out_loc` are used in
  `scheduler.py`. This is the hardest part for most people.
- **Compare radix vs naive cache.** Diff `radix_cache.py` against
  `naive_cache.py`. Then ask: why is the naive one even kept? (Answer:
  lock-free, simpler, sometimes faster on workloads with no prefix sharing.)

---

## Credit

- Upstream repo: [`sgl-project/mini-sglang`](https://github.com/sgl-project/mini-sglang)
- Concepts: SGLang, vLLM (PagedAttention), Sarathi-Serve (chunked prefill),
  NanoFlow (overlap scheduling), FlashAttention, FlashInfer, TensorRT-LLM.
