# Stage 0 — Native Metal Benchmark (M3 Max)

**Machine:** Apple M3 Max (`gpu,t6031`, 40-core GPU), 48 GB unified memory, macOS
**Runtime:** Ollama 0.30.10 native (Metal), Q4 GGUF, temp 0.1, num_predict 256
**Condition:** clean (background apps present — machine was under load, ~4.5 GB swap at baseline)
**Harness:** `sandbox/harness/stage0_bench.py`, label `metal-clean`, uncapped (`iogpu.wired_limit_mb=0`)

## Summary of findings

**Recommendation: qwen2.5:3b (Q4_K_M) as the primary model — confirmed.** This M3 Max
study, combined with the team's earlier base-M3 (Ollama) and Windows RTX 2070 (LM Studio)
runs, gives three independent Stage 0 evaluations that all converge on qwen2.5:3b.

1. **VRAM headroom is the gating metric, and qwen wins it decisively.** qwen wires only
   ~2.6 GB and stays fully GPU-resident at full speed down to a simulated **4 GB** ceiling;
   it is the *only* 3–4B candidate that does. phi3.5 already needs CPU help at 8 GB
   (~8.5 GB footprint), gemma and llama sit in between.

2. **Raw speed is hardware-specific and must not drive the choice.** Uncapped, the M3 Max
   ranks phi3.5 (108) ≳ llama (101) > qwen (92) > gemma (73) tok/s — the *reverse* of the
   base-M3 ranking. tokens/sec does not transfer between GPUs; the VRAM-capacity ranking does.

3. **Strict-JSON reliability separates qwen from the pack.** qwen emits clean JSON arrays;
   gemma and phi wrap them in markdown fences and llama is messy — and strict JSON is central
   to the Stage 2 instruction-pair stage.

4. **The size spectrum brackets the decision** (see Appendix for the sub-1B sweep):

   | Tier | VRAM problem? | Quality problem? |
   | ---- | ------------- | ---------------- |
   | sub-1B (qwen 0.5b / gemma 270m / llama 1b) | ✅ solved — fits any GPU, 2–3× faster | ❌ all fail strict-JSON; qwen-0.5b drifts to Chinese |
   | **qwen2.5:3b (chosen)** | ✅ fits to 4 GB | ✅ clean strict-JSON |
   | 4B+ (gemma 4b / phi 3.8b) | ❌ phi needs >8 GB | mixed |

   Going smaller solves the hardware constraint but breaks the quality requirement; 3B is the
   sweet spot that clears both bars.

**Caveats (don't over-read the numbers):**
- The simulated VRAM cap (`iogpu.wired_limit_mb`) bounds GPU *wired memory* but doesn't tell
  Ollama's offload planner to use fewer layers, so under a constrained partial GPU/CPU split it
  over-commits and emits empty output (gemma@4 GB; qwen/llama/phi@2 GB; llama-1b@2 GB). This
  **overstates** non-qwen failures; a real GPU of that size would offload-and-run slowly. The
  capacity *ranking* and qwen's clean fit are unaffected.
- Apple-GPU *speed* does not transfer to a discrete GPU; only the VRAM *capacity* picture does.
- **Still outstanding:** no run yet on the true deployment target (16 GB RAM + discrete NVIDIA
  GPU on Windows). The sandbox's Docker CUDA mode is built to capture it on real hardware.

## Consolidated comparison — all models × all VRAM ceilings

The single side-by-side view. Detail and methodology for each condition follow below.

**Tokens/sec by simulated VRAM ceiling:**

| Model | Uncapped | 8 GB | 4 GB | 2 GB |
| ----------- | :------: | :--: | :--: | :--: |
| **qwen2.5:3b** | 91.7 | 99.0 | **100.1** | —\* |
| gemma3:4b | 73.0 | 78.9 | —\* | 22.7 |
| llama3.2:3b | 100.6 | 109.1 | **9.9** | —\* |
| phi3.5 | 108.1 | **9.2** | ~20 | —\* |

**Model VRAM allocated (GB) — shows *why* throughput holds or collapses:**

| Model | Uncapped | 8 GB | 4 GB | 2 GB |
| ----------- | :------: | :--: | :--: | :--: |
| qwen2.5:3b | 2.60 | 2.60 | 2.60 | 2.14 |
| gemma3:4b | 3.54 | 3.54 | 3.49 | 0.12 |
| llama3.2:3b | 3.92 | 3.92 | 3.39 | 2.33 |
| phi3.5 | 8.53 | 7.22 | 2.90 | 1.30 |

\* `—` = the model returned **empty output (0 tokens generated)** for those prompts —
a real generation failure, not a measurement gap. (Verified: responses are empty
strings, so no `eval_count` is reported.)

**Why the failures cluster in the "middle":** these failures all occur in a *partial*
GPU+CPU offload state, and they expose a limitation of this cap method.
`iogpu.wired_limit_mb` caps GPU **wired memory** but does **not** inform Ollama's
layer-offload planner of a reduced VRAM budget — Ollama still sees the full 48 GB
unified pool, over-commits layers to the GPU, then fails to allocate working memory at
runtime and emits nothing. A model is fine when it either (a) fully fits on GPU, or
(b) falls back to ~pure CPU; it breaks in the awkward hybrid between. A **real**
discrete GPU of the same size behaves better: Ollama reads the true VRAM size up front
and offloads proactively, yielding slow-but-working output rather than empty output.

**What still holds:** qwen keeps its full 2.6 GB GPU allocation and full speed down to
4 GB — it genuinely fits — while phi3.5 already needs CPU help at 8 GB. The capacity
**ranking** (qwen < llama < gemma < phi) is real and transfers. **What is overstated:**
the empty-output collapses of the non-qwen models are partly an artifact of the
wired-limit method; for a faithful per-size prediction, also constrain Ollama's offload
(`num_gpu` layer count / a VRAM-budget env) instead of only the wired limit.
(Speed itself is M3-Max-specific and never transfers to a discrete GPU.)

## Results (uncapped detail)

| Model | tok/s (mean) | Model VRAM (GB)¹ | Peak GPU mem (MB)² | Load footprint (GB)³ | Strict JSON | 5-bullet format |
| ----------- | :----------: | :--------------: | :----------------: | :------------------: | :---------: | :-------------: |
| qwen2.5:3b  |     91.7     |     **2.60**     |       5 010        |        1.89          | ✅ clean array | ⚠️ dash on own line |
| gemma3:4b   |     73.0     |       3.54       |       6 907        |        4.07          | ❌ ```json fence | ✅ clean |
| llama3.2:3b |    100.6     |       3.92       |       6 145        |        2.95          | ✅ (borderline) | ❌ `- -` doubled |
| phi3.5      |   **108.1**  |     **8.53**     |      11 099        |        5.49          | ❌ ```json fence | ✅ clean |

¹ Ollama `size_vram` — model weights + KV cache wired to the GPU.
² `IOAccelerator` "In use system memory" peak via `ioreg` (system-wide GPU mem).
³ Used-memory delta from baseline; absolute system RAM was environment-inflated and is not a clean signal here.

Per-prompt tok/s was stable (±3) for every model, so these means are reliable.

## Key findings

1. **Speed ranking flips vs the base-M3 / Windows runs.** Here phi3.5 (108) ≳ llama (101) > qwen (92) > gemma (73). On Kirill's *base* M3 it was qwen (43) > llama (42) > phi (38) > gemma (33). The M3 Max's 40-core GPU is ~2.0–2.7× faster *and* reorders the field — concrete proof that tok/s is hardware-specific and does not transfer.

2. **VRAM headroom — the decisive axis — still favours qwen.** qwen wires just **2.6 GB**; phi3.5 balloons to **8.53 GB** (KV cache from its large default context), llama 3.92, gemma 3.54. Under the target's **2–8 GB VRAM**, phi3.5 would *not* fit an 8 GB card; qwen has the most headroom by far.

3. **Strict-JSON: qwen is the only clean emitter among the strong options.** gemma and phi wrap output in ```json fences (fail strict-JSON-only — matches the team's Windows report); llama is borderline/messy. Strict JSON is central to Stage 2 instruction-pair generation.

4. **Net: the M3 Max data reinforces qwen2.5:3b as primary.** The two *faster* models (phi, llama) each lose on the metrics that actually gate this project — VRAM headroom and strict-JSON reliability. Being mid-pack on raw speed on one GPU doesn't change the selection.

## Caveats

- **Speed is not transferable** to the target's discrete GPU (different cores/bandwidth). Capacity ("fits in N GB VRAM?") *is* reproducible via `iogpu.wired_limit_mb`.
- Absolute system RAM/swap was inflated by other running apps; treat Model VRAM and tok/s as the clean signals.
- phi3.5's 8.5 GB is mostly KV cache (large default context) — could be cut with a smaller `num_ctx`. Worth testing before disqualifying it.

## 8 GB VRAM-cap pass (`metal-cap8`, `iogpu.wired_limit_mb=8192`)

Simulates the target's discrete-VRAM ceiling on the M3 Max by capping GPU-wired
unified memory to 8 GB. Tests the headroom question directly.

| Model | Uncapped tok/s | Capped tok/s | Δ | Verdict @ 8 GB |
| ----------- | :------------: | :----------: | :------: | :-- |
| qwen2.5:3b  |      91.7      |     99.0     |   +8%    | ✅ fits easily (2.6 GB) |
| gemma3:4b   |      73.0      |     78.9     |   +8%    | ✅ fits (3.5 GB) |
| llama3.2:3b |     100.6      |    109.1     |   +8%    | ✅ fits (3.9 GB) |
| phi3.5      |     108.1      |   **9.2**    | **−91%** | ❌ collapses — spills to CPU |

**Finding:** phi3.5 — the *fastest* model uncapped — drops ~12× (108→9.2 tok/s)
under an 8 GB ceiling because its weights + KV cache don't fit, forcing CPU
offload. On a ≤8 GB target GPU it is effectively unusable unless `num_ctx` is cut
hard. qwen/gemma/llama already fit under 8 GB, so the cap doesn't affect them (the
+8% is run-to-run/environmental noise, not a cap effect).

**Conclusion:** all three Stage 0 runs — base-M3 (Ollama), Windows RTX 2070 (LM
Studio), and M3 Max + simulated 8 GB cap — converge on **qwen2.5:3b** as primary.
The capped pass adds the VRAM-headroom data point the earlier Apple-Silicon run
could not produce.

## 4 GB VRAM-cap pass (`metal-cap4`, low-end GPU class)

| Model | Uncapped | 8 GB cap | 4 GB cap | Behavior at 4 GB |
| ----------- | :------: | :------: | :------: | :-- |
| qwen2.5:3b  |   91.7   |   99.0   | **100.1**| ✅ fully GPU-resident, untouched |
| gemma3:4b   |   73.0   |   78.9   | degraded | spill (3.5 GB GPU + 3 GB CPU); no clean tok/s recorded |
| llama3.2:3b |  100.6   |  109.1   |   9.9    | ❌ collapsed ~10× (CPU spill) |
| phi3.5      |  108.1   |    9.2   |  ~20     | ❌ collapsed (10.8 GB total footprint) |

**Finding:** at a 4 GB VRAM budget, **qwen2.5:3b is the only viable model** — and it
doesn't even slow down. Every other candidate spills to CPU and collapses (gemma so
badly it returned no clean throughput). Tiering:

| GPU VRAM class | Viable candidates |
| -------------- | ----------------- |
| ≥ 8 GB | qwen, llama, gemma (phi ❌) |
| 4 GB (low-end) | **qwen only** |

(Harness note: under heavy CPU spill gemma3 returned generations without eval stats,
so its 4 GB throughput shows as null rather than a number — a measurement gap, but the
degraded behavior is unambiguous from the GPU/CPU split.)

## 2 GB VRAM-cap pass (`metal-cap2`, the floor)

At 2 GB, after the ~1.5 GB the display compositor already holds, there is essentially
no GPU budget left for a model. Ollama force-spilled all four to CPU.

| Model | uncap | 8 GB | 4 GB | 2 GB | GPU mem @ 2 GB |
| ----------- | :---: | :--: | :--: | :--: | :------------: |
| qwen2.5:3b  | 91.7  | 99.0 |100.1 | (no stats) | 2.14 GB |
| gemma3:4b   | 73.0  | 78.9 | deg. | 22.7  | 0.12 GB (≈all CPU) |
| llama3.2:3b |100.6  |109.1 |  9.9 | (no stats) | 2.33 GB |
| phi3.5      |108.1  |  9.2 | ~20  | (no stats) | 1.3 GB |

**Finding:** no candidate stays GPU-resident at 2 GB; throughput drops to the CPU-only
floor (~20 tok/s, seen cleanly on gemma which ran almost entirely on CPU). 3 of 4
returned no `eval_count`/`eval_duration` under extreme spill — a **harness measurement
gap**, not a model failure, so the 2 GB tok/s are not trustworthy. The reliable takeaway:
**4 GB is the practical lower bound for GPU-resident operation, and only qwen clears it.**
Below 4 GB everything is CPU-bound regardless of model.

## Appendix — Small-tier sweep (sub-1B / smallest-of-family)

Tested the smallest available model in each family under the same uncapped + 8/4/2 GB
conditions. Only qwen2.5 ships a true 0.5B; gemma's smallest is 270m (0.27B), llama's
is 1b, and phi3.5 has no variant below 3.8B (excluded). Labels: `metal-small-*`.

**Tokens/sec by VRAM ceiling:**

| Model | Uncapped | 8 GB | 4 GB | 2 GB |
| --------------- | :------: | :--: | :--: | :--: |
| qwen2.5:0.5b | 219 | 223 | 217 | 218 |
| gemma3:270m | 234 | 235 | 230 | 236 |
| llama3.2:1b | 174 | 172 | 167 | — (1/5 resp) |

**Model VRAM (GB):** qwen2.5:0.5b 0.69 · gemma3:270m 0.38 · llama3.2:1b 1.93.

**Findings:**
- At sub-1B, **VRAM headroom is no longer the constraint** — qwen2.5:0.5b and gemma3:270m
  run at full speed across *every* ceiling including 2 GB (they're <0.7 GB). Only
  llama3.2:1b broke at 2 GB, and that's the same wired-limit partial-offload artifact
  (1.93 GB model + ~1.5 GB display baseline > 2 GB), not a real "won't fit 2 GB".
- **Speed is 2–3× the 3–4B tier** (174–235 vs 73–108 tok/s).
- **Quality regresses to disqualifying:** all three fail strict-JSON (markdown fences),
  and qwen2.5:0.5b answered the strict-JSON prompt **in Chinese** (language-consistency
  failure). For a Stage-2 pipeline that depends on clean strict-JSON, sub-1B is too weak.

**Conclusion:** going smaller solves the hardware constraint but breaks the quality
requirement. qwen2.5:**3b** remains the sweet spot — fits a 4 GB GPU *and* emits clean
strict-JSON. 0.5B/sub-1B is not viable for this use case.
