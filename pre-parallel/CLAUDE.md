# pre-parallel — CUDA MNIST FFN

A feedforward MNIST classifier being progressively parallelized for a parallel
computing class. Graded on Darwin against fixed hyperparameters in `config.h`.

## Architecture

- 784 (input) → **256** (ReLU) → **128** (ReLU) → 10 (softmax)
- Loss: categorical cross-entropy
- Optimizer: vanilla SGD (currently per-sample, batch size 1)
- `EPOCHS=5`, `NUM_TRAIN=60000`, `LR=0.01f`

> The header comment at the top of `nnp.cu` still says `128 → 64`. That is stale —
> trust `config.h` (`H1=256`, `H2=128`). Fix the comment when you're next editing
> the file.

## Layout conventions

All matrices are **row-major**. Weights `W` have shape `[in_dim, out_dim]` and
are indexed `W[i * out_dim + j]`. Activations are row vectors, so the forward
op is `out = in @ W + b`. For batched ops with batch size `B`:

- Activations: `[B, dim]` (row-major, sample-major)
- Forward: `Out[B, out] = In[B, in] @ W[in, out] + b` (bias broadcast over `B`)
- Backward error prop: `delta_prev[B, in] = delta[B, out] @ W^T[out, in]`
- Weight gradient: `dW[in, out] = In^T[in, B] @ delta[B, out]` (sum over batch)
- Bias gradient: `db[out] = sum over B of delta[B, out]`

## Files & "do not touch"

| File | Status |
| --- | --- |
| `nnp.cu` | edit (training loop, kernel launches, graph capture) |
| `kernels.cu` / `kernels.h` | edit (CUDA kernels) |
| `makefile` | edit (link `-lcublas` for Phase 2) |
| `config.h` | **DO NOT EDIT** — graded against fixed values on Darwin |
| `nnp.h` | do not touch |
| `loader.cc` / `loader.h` | do not touch |
| `main.cc` | do not touch |

`predict()` in `nnp.cu` is CPU-only; leave it alone — it's not on the hot path.

> Note: `config.h:28` already defines `BATCH 64`. The original plan was to put
> `B` in `kernels.h` or `nnp.cu` instead, but since `BATCH` is already there
> and is not one of the values the autograder is sensitive to, leave it.
> Reference it as `BATCH` from `nnp.cu` and `kernels.cu`.

## Build & run

From `pre-parallel/`:

```
make            # produces ./nnp
./nnp train     # trains, writes model.bin, prints per-epoch loss + wall time
./nnp predict   # loads model.bin, prints a predicted digit
```

## Current state (already done)

- Forward, backprop, deltas, weight/bias updates all run on the GPU
  (`kernels.cu` + launch sites in `nnp.cu:131-156`).
- Per-sample loss is accumulated on-device via `accumulate_loss` — no per-sample
  device-to-host copy.
- The per-sample kernel sequence is captured into a CUDA graph
  (`cudaStreamBeginCapture` / `cudaStreamEndCapture` at `nnp.cu:129-159`) and
  replayed via `cudaGraphLaunch` once per sample. Inputs are routed through
  fixed `d_input` / `d_label` scratch buffers so the graph's pointers don't
  change across iterations.
- Wall time history: 25s → 23s → 18s → **1.2s** (Phase 1) → **1.00s** (Phase 2). Goal hit.

## Verification protocol (run after every phase)

1. `make` builds clean (no warnings introduced).
2. `time ./nnp train` — record wall time; loss curve must trend downward across
   the 5 epochs and look sane vs. the previous run (see per-phase notes for the
   expected shape — it will *not* match the previous run numerically once
   batching changes update semantics).
3. `./nnp predict` runs without crashing and prints a plausible digit
   (0–9, confidence > random).
4. Note the new wall time in this file when a phase lands.

---

## Phase 1 — Mini-batch SGD with custom batched kernels

Target: **~3–5s** wall time for `./nnp train`.

### Decisions already made

- **`B = 64`** (already `#define BATCH 64` in `config.h`). Don't re-decide.
- The autograder is sensitive to `LR`, `EPOCHS`, `NUM_TRAIN`, architecture
  dims — *not* batch size. So fold the `1/B` factor into the update kernels
  (`W += (LR/B) * input^T @ delta`) rather than editing `LR` in `config.h`.

### Kernel changes

Every kernel gains a batch dimension:

- `matvec_relu`, `matvec_softmax` → batched matmul + bias-broadcast + activation.
  Output is `[B, out_dim]`. Softmax is per-row (per-sample).
- `compute_delta3` → operate per-sample across the batch; output `[B, CLASSES]`.
- `compute_delta_hidden` → same, output `[B, dim]`. The `h_act > 0` mask is
  per-element of the batched activation tensor.
- `weight_update` → becomes `dW = input^T @ delta` (a matmul, summed over the
  batch) followed by `W += (LR/B) * dW`. You can either do this as two kernels
  or fuse the scale into the accumulation.
- `bias_update` → reduce `delta` along the batch dim, then `b += (LR/B) * db`.
- `accumulate_loss` → sum per-sample CE across the batch and atomic-add to
  the epoch accumulator.

### Training loop

Inner loop in `nnp.cu` becomes:

```c
for (int batch_start = 0; batch_start < NUM_TRAIN; batch_start += BATCH) {
    // memcpyAsync BATCH samples into d_input ([BATCH, SIZE])
    // memcpyAsync BATCH labels  into d_label ([BATCH, CLASSES])
    cudaGraphLaunch(graph_exec, stream);   // re-captured over batched ops
}
```

`NUM_TRAIN=60000` is divisible by 64? No — `60000 / 64 = 937` full batches +
32 leftover. **Drop the final partial batch** (simplest; loses 32/60000 of an
epoch which is fine) or pad it. Document the choice when you implement.

### CUDA graph

Re-capture the graph over the per-batch sequence. With `5 * 937 ≈ 4685`
iterations instead of 300k, launch overhead is much less significant —
benchmark with and without the graph and keep whichever is faster. If you
remove it, also remove the `d_input`/`d_label` scratch routing.

### Expected loss-curve shape

Mini-batch SGD with `(LR/B)` scaling gives **smoother but slower-descending**
loss than per-sample SGD at the same nominal `LR`. After 5 epochs the final
loss will likely be a bit higher than the B=1 run — that is expected and not
a regression. As long as the curve is monotone-ish downward and `./nnp predict`
still gives plausible digits, Phase 1 is good. Don't let the next phase mistake
this for a bug.

---

## Phase 2 — cuBLAS GEMMs

Target: **~1s** wall time.

### Setup

- Add `-lcublas` to the link line in `makefile`.
- `#include <cublas_v2.h>` in `nnp.cu`.
- In `train_model`, create a `cublasHandle_t`, then bind it to the capture
  stream with `cublasSetStream(handle, stream)` *before* `cudaStreamBeginCapture`.
  cuBLAS calls inside the captured region will then be recorded into the graph.

### Replace the three matmul-shaped operations with `cublasSgemm`

1. **Forward:** `Out[B, out] = In[B, in] @ W[in, out] + b`
2. **Backward error propagation:** `delta_prev[B, in] = delta[B, out] @ W^T[out, in]`
3. **Weight gradient:** `dW[in, out] = In^T[in, B] @ delta[B, out]`

cuBLAS is column-major. The standard trick for row-major `C = A @ B` is to
call `cublasSgemm` as if computing `C^T = B^T @ A^T` — pass the row-major
pointers as-is, swap operand order, and use the row-major dimensions as the
column-major shape of the transposed result. Decide on a consistent helper
(macro or inline wrapper) and reuse it for all three GEMMs.

For the weight update, you can either:
- Use `cublasSgemm` to compute `dW`, then a small custom kernel to apply
  `W += (LR/B) * dW` (clean), or
- Use `cublasSgemm` with `beta = 1.0` and `alpha = LR/B` to fuse the apply
  step directly into the accumulation against `W` (one fewer kernel). Prefer
  this if it works cleanly — fewer launches inside the graph.

### Keep as custom kernels

Activations (ReLU, softmax), bias broadcast/reduce, `compute_delta3`,
`compute_delta_hidden`'s ReLU-mask step, `accumulate_loss`. cuBLAS doesn't
help with these and they're cheap.

### Verification

Same protocol as Phase 1. Loss curve should match Phase 1's shape closely
(numerics will differ slightly due to GEMM accumulation order, but per-epoch
losses should be within a few percent). If `./nnp predict` regresses,
suspect a transpose / leading-dim mismatch in one of the three GEMMs — that's
the #1 failure mode here.

### Expected outcome

Wall time around **1s**. If you land at 1.5–2s, that's still good; further
gains likely require fusing bias+activation into the GEMM epilogue or moving
to Tensor Cores, which is past the scope of this assignment.
