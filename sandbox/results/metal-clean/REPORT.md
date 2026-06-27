# Stage 0 — Native Metal Benchmark (M3 Max)

**Machine:** Apple M3 Max (`gpu,t6031`, 40-core GPU), 48 GB unified memory, macOS
**Runtime:** Ollama 0.30.10 native (Metal), Q4 GGUF, temp 0.1, num_predict 256
**Condition:** clean (background apps present — machine was under load, ~4.5 GB swap at baseline)
**Harness:** `sandbox/harness/stage0_bench.py`, label `metal-clean`, uncapped (`iogpu.wired_limit_mb=0`)

## Summary of findings

**Recommendation: qwen2.5:3b (Q4_K_M) as the primary model — confirmed.** This M3 Max
study, combined with the team's earlier base-M3 (Ollama) and Windows RTX 2070 (LM Studio)
runs, gives three independent Stage 0 evaluations that all converge on qwen2.5:3b.

1. **VRAM headroom is the gating metric, and qwen wins it.** qwen wires only ~2.6 GB and
   stays fully GPU-resident at full speed down to a simulated **4 GB** budget. Under the
   *faithful* `num_gpu`-offload method (see "Faithful VRAM-budget sweep"), all four models
   still *run* at every budget down to 2 GB — a real GPU offloads gracefully rather than
   failing — but qwen degrades the least: at 2 GB it keeps 28/36 layers on GPU and leads on
   throughput (61 vs 33–37 tok/s), because it is the smallest. phi3.5 is the first to need
   offload (it is the largest, ~8.5 GB). NOTE: the earlier wired-limit "collapse/empty"
   results (e.g. phi@8 GB) were a method artifact — corrected by the faithful sweep.

2. **Raw speed is hardware-specific and must not drive the choice.** Uncapped, the M3 Max
   ranks phi3.5 (108) ≳ llama (101) > qwen (92) > gemma (73) tok/s — the *reverse* of the
   base-M3 ranking. tokens/sec does not transfer between GPUs; the VRAM-capacity ranking does.

3. **Strict-JSON reliability separates qwen from the pack, but it is not perfect.** In the
   full-response quality rerun, qwen2.5:3b was the only 3–4B model to pass strict JSON at all
   (2/4 conditions). gemma, llama, phi, and every sub-1B/smallest-tier model scored 0/4.
   Stage 2 should still validate JSON and retry/repair failures rather than trusting raw output.

4. **The size spectrum brackets the decision** (see Appendix for the sub-1B sweep):

   | Tier | VRAM problem? | Quality problem? |
   | ---- | ------------- | ---------------- |
   | sub-1B (qwen 0.5b / gemma 270m / llama 1b) | ✅ solved — fits any GPU, 2–3× faster | ❌ all fail strict-JSON; qwen-0.5b drifts to Chinese |
   | **qwen2.5:3b (chosen)** | ✅ fits to 4 GB | best strict-JSON result (2/4), needs validation/retry |
   | 4B+ (gemma 4b / phi 3.8b) | ❌ phi needs >8 GB | mixed |

   Going smaller solves the hardware constraint but breaks the quality requirement; 3B is the
   sweet spot that clears both bars.

5. **Quality degrades more from shrinking parameter count than from shrinking VRAM.** Under
   the full-response quality rerun, all models produced non-empty responses at every budget.
   VRAM reduction mainly reduced throughput; parameter reduction caused stronger structured
   output regressions. qwen2.5:0.5b gained speed but scored 0/4 on strict JSON and drifted into
   Chinese in 3/4 strict-JSON conditions.

**Caveats (don't over-read the numbers):**
- Two VRAM-constraint methods are in this report. The **`iogpu.wired_limit_mb` caps**
  (8/4/2 GB sections) over-commit and emit empty output under constraint — a method
  artifact. The **`num_gpu`-offload "Faithful VRAM-budget sweep"** is the trustworthy one
  and supersedes it for "would it run / how fast": there, every model runs at every budget
  and qwen degrades the least. Prefer the faithful numbers; the wired-limit sections are
  retained for transparency.
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
| gemma3:4b   |     73.0     |       3.54       |       6 907        |        4.07          | ❌ markdown JSON fence | ✅ clean |
| llama3.2:3b |    100.6     |       3.92       |       6 145        |        2.95          | ✅ (borderline) | ❌ `- -` doubled |
| phi3.5      |   **108.1**  |     **8.53**     |      11 099        |        5.49          | ❌ markdown JSON fence | ✅ clean |

¹ Ollama `size_vram` — model weights + KV cache wired to the GPU.
² `IOAccelerator` "In use system memory" peak via `ioreg` (system-wide GPU mem).
³ Used-memory delta from baseline; absolute system RAM was environment-inflated and is not a clean signal here.

Per-prompt tok/s was stable (±3) for every model, so these means are reliable.

## Key findings

1. **Speed ranking flips vs the base-M3 / Windows runs.** Here phi3.5 (108) ≳ llama (101) > qwen (92) > gemma (73). On Kirill's *base* M3 it was qwen (43) > llama (42) > phi (38) > gemma (33). The M3 Max's 40-core GPU is ~2.0–2.7× faster *and* reorders the field — concrete proof that tok/s is hardware-specific and does not transfer.

2. **VRAM headroom — the decisive axis — still favours qwen.** qwen wires just **2.6 GB**; phi3.5 balloons to **8.53 GB** (KV cache from its large default context), llama 3.92, gemma 3.54. Under the target's **2–8 GB VRAM**, phi3.5 would *not* fit an 8 GB card; qwen has the most headroom by far.

3. **Strict-JSON: qwen is the only candidate that passes at all in the full-response audit.** qwen2.5:3b passes strict JSON in 2/4 full-response conditions; gemma, llama, phi, and the small-tier models score 0/4. Strict JSON is central to Stage 2 instruction-pair generation, so the pipeline should include parse validation and retry/repair even with qwen.

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

## Faithful VRAM-budget sweep (num_gpu offload) — PREFERRED over the wired-limit caps

The cap passes above used `iogpu.wired_limit_mb`, which (as their footnote warns) makes
Ollama over-commit and emit empty output under constraint. This sweep instead sets
Ollama's `num_gpu` per model — the *supported* mechanism a real VRAM-limited GPU triggers,
offloading surplus layers to CPU. For each model we measured its full GPU footprint and
layer count, mapped each budget to a layer count, and reran. Labels: `metal-budget{8,4,2}`.
All runs produced 5/5 non-empty responses.

**Tokens/sec — and (layers kept on GPU):**

| Model | 8 GB | 4 GB | 2 GB |
| ----------- | :------------: | :------------: | :------------: |
| qwen2.5:3b  | 86.9 (36/36)   | 87.8 (36/36)   | **61.3 (28/36)** |
| gemma3:4b   | 69.7 (34/34)   | 71.4 (34/34)   | 33.0 (19/34)   |
| llama3.2:3b | 93.8 (28/28)   | 93.1 (28/28)   | 37.2 (14/28)   |
| phi3.5      | 91.8 (30/32)   | 49.6 (15/32)   | 33.5 (8/32)    |

![Faithful VRAM-budget degradation for the 3-4B models: phi3.5 and llama3.2 start fastest but fall steeply as the budget tightens, while qwen2.5:3b declines least and is highest at the 2 GB budget.](../../charts/degradation_3to4b.svg)

**This corrects three artifacts of the wired-limit method:**
- **phi3.5 @ 8 GB does NOT collapse.** Faithful: ~92 tok/s (only 2 of 32 layers offloaded).
  The wired-limit "9.2 tok/s collapse" was a method artifact, not real behaviour.
- **llama3.2 @ 4 GB is fine** (~93 tok/s, fully resident — its 3.8 GB fits 4 GB). The
  wired-limit "9.9" was an artifact.
- **Every "EMPTY" cell runs** (gemma@4 GB, qwen/llama/phi@2 GB) at 33–61 tok/s.

**The honest, corrected reading:** down to a 2 GB budget, *all four models run* — a real
GPU offloads gracefully rather than failing. The differentiator is **how gracefully each
degrades**, and qwen still wins it: at the tight 2 GB budget qwen keeps the most layers on
GPU (28/36 ≈ 78%) and the highest throughput (61 tok/s vs 33–37 for the rest), because it
is the smallest. phi3.5 is the first to need offload (already at 8 GB). So the selection
conclusion holds, but on "degrades most gracefully / stays fastest under constraint"
rather than "others catastrophically fail." (Speed remains M3-Max-specific; the relative
ordering is the transferable signal.)

## Quality degradation: VRAM budget vs parameter size

This section uses a second full-response rerun, not the original 200-character previews.
Labels: `metal-quality-clean`, `metal-quality-budget{8,4,2}`,
`metal-small-quality-clean`, and `metal-small-quality-budget{8,4,2}`.

The harness stored the complete response for every prompt and computed conservative,
parser-oriented checks:

- non-empty response;
- exact five dash bullets for the bullet prompt;
- strict JSON parse with exactly two `{instruction, response}` objects;
- grounded-answer keyword check;
- policy/procedure sentence-count check;
- honest-unknown check;
- markdown-fence and CJK/Chinese-character detection.

These checks are intentionally stricter than "sounds reasonable"; they approximate whether the
pipeline can consume the output automatically.

### A. Decreasing VRAM budget: no empty-output collapse, but speed drops

Using faithful `num_gpu` offload, every 3–4B model produced full responses in every condition.
The main degradation from lower VRAM is throughput. Structured-output reliability is model
specific and somewhat run-variable, but it does **not** turn into the empty-output failure seen
with the wired-limit method.

| Model | Quality-check passes¹ | Non-empty responses | Strict JSON pass² | Exact 5 bullets | tok/s uncapped → 2 GB | Speed change |
| ----------- | :------------------: | :-----------------: | :---------------: | :-------------: | :-------------------: | :----------: |
| qwen2.5:3b | 8/20 | 20/20 | **2/4** | 0/4 | 91.9 → 43.7 | −52% |
| gemma3:4b | 16/20 | 20/20 | 0/4 | **4/4** | 74.7 → 23.1 | −69% |
| llama3.2:3b | 12/20 | 20/20 | 0/4 | **4/4** | 97.2 → 27.4 | −72% |
| phi3.5 | 7/20 | 20/20 | 0/4 | 1/4 | 102.1 → 26.7 | −74% |

¹ Five deterministic checks per condition × four conditions = 20.
² Strict JSON counted across uncapped, 8 GB, 4 GB, and 2 GB full-response runs.

Important nuance: qwen2.5:3b passed strict JSON in the uncapped and 8 GB rerun, but failed at
4 GB and 2 GB. The 4 GB case is still fully GPU-resident (`36/36` layers), so this should be
read as structured-output fragility/run variation, not proof that 4 GB VRAM directly damages
semantics. The practical conclusion is still clear: qwen is the only strong candidate that
passes strict JSON at all, but Stage 2 must include JSON validation plus retry/repair.

![Strict-JSON pass rate across the four VRAM conditions: qwen2.5:3b passes 2 of 4, while gemma3:4b, llama3.2:3b and phi3.5 each pass 0 of 4.](../../charts/strict_json_passes.svg)

### B. Decreasing parameter size: structured output gets worse despite easy fit

The small-tier models fit easily in memory and remain fast, but the full-response audit shows
that speed comes with weaker output discipline.

| Family | Strong model strict JSON | Small model strict JSON | Small model CJK drift | Quality passes: strong → small | tok/s: strong → small | Speed change |
| ------ | :----------------------: | :---------------------: | :-------------------: | :----------------------------: | :-------------------: | :----------: |
| qwen | **2/4** (`qwen2.5:3b`) | 0/4 (`qwen2.5:0.5b`) | **3/4** | 8/20 → 6/20 | 91.9 → 220.2 | +140% |
| gemma | 0/4 (`gemma3:4b`) | 0/4 (`gemma3:270m`) | 0/4 | 16/20 → 8/20 | 74.7 → 216.0 | +189% |
| llama | 0/4 (`llama3.2:3b`) | 0/4 (`llama3.2:1b`) | 0/4 | 12/20 → 8/20 | 97.2 → 157.1 | +62% |
| phi | 0/4 (`phi3.5`) | N/A | N/A | N/A | No smaller Ollama variant | N/A |

All three small-tier models produced 20/20 non-empty responses and all stayed fully
GPU-resident even at the 2 GB `num_gpu` budget. Their failure mode is therefore not hardware
fit; it is structured-output competence. The qwen family shows the tradeoff most clearly:
`qwen2.5:0.5b` is ~2.4× faster than `qwen2.5:3b`, but drops from 2/4 to 0/4 strict-JSON
passes and drifts into Chinese in 3/4 strict-JSON conditions.

**Conclusion:** VRAM reduction and parameter reduction fail differently. Lower VRAM, when
handled with `num_gpu`, mainly reduces speed and does not cause empty generations. Lower
parameter count preserves speed and memory fit but damages structured output. qwen2.5:3b
remains the best compromise because it is the only model that ever passes strict JSON in the
full-response audit while still fitting low-end VRAM, but the pipeline should treat raw model
JSON as untrusted and validate/retry it.

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

**Faithful `num_gpu` budget check (full curve, tok/s and layers-on-GPU):** the small-tier
models stay fully GPU-resident through 2 GB, so the sweep was extended to **1 GB and
0.5 GB** to find where they actually start offloading. Labels:
`metal-small-budget{8,4,2,1,05}`.

| Model | 8 GB | 4 GB | 2 GB | 1 GB | 0.5 GB |
| --------------- | :----------: | :----------: | :----------: | :----------: | :----------: |
| qwen2.5:0.5b | 195 (24/24) | 192 (24/24) | 195 (24/24) | 185 (24/24) | 181 (17/24) |
| gemma3:270m | 208 (18/18) | 197 (18/18) | 207 (18/18) | 197 (18/18) | 193 (18/18) |
| llama3.2:1b | 135 (16/16) | 140 (16/16) | 140 (16/16) | **71 (8/16)** | **53 (4/16)** |

![Faithful VRAM-budget degradation for the sub-1B tier from 8 GB down to 0.5 GB: gemma3:270m stays flat (never offloads), qwen2.5:0.5b barely dips at 0.5 GB, and llama3.2:1b halves at 1 GB and drops further at 0.5 GB while still producing output.](../../charts/degradation_small.svg)

All runs produced 5/5 non-empty responses at every budget — including 0.5 GB — confirming
the faithful method shows **no failures anywhere** (the wired-limit `llama3.2:1b @ 2 GB`
"failure" was a cap artifact). The degradation curve:
- **gemma3:270m** (0.38 GB): never offloads, full speed even at 0.5 GB — effectively VRAM-free.
- **qwen2.5:0.5b** (0.69 GB): full through 1 GB; at 0.5 GB it offloads 7/24 layers yet barely
  slows (181 tok/s) — still ~2× the 3B tier's uncapped speed.
- **llama3.2:1b** (1.93 GB): the only one that meaningfully degrades — halves at 1 GB
  (71 tok/s, 8/16 layers) and ~53 tok/s at 0.5 GB, but still produces full working output.

So even at a 0.5 GB budget all three run; memory is a non-issue at this tier. The conclusion
does not change: memory is solved sub-1B, but output quality is not.

**Findings:**
- At sub-1B, **VRAM headroom is no longer the constraint** — qwen2.5:0.5b and gemma3:270m
  run at full speed across *every* ceiling including 2 GB (they're <0.7 GB). Only
  llama3.2:1b broke at 2 GB, and that's the same wired-limit partial-offload artifact
  (1.93 GB model + ~1.5 GB display baseline > 2 GB), not a real "won't fit 2 GB".
- **Speed is 2–3× the 3–4B tier** (174–235 vs 73–108 tok/s).
- **Quality regresses to disqualifying:** the full-response quality rerun confirms all three
  small-tier models score **0/4 strict-JSON passes**. qwen2.5:0.5b also drifts into Chinese
  in **3/4** strict-JSON conditions. For a Stage-2 pipeline that depends on parser-safe JSON,
  sub-1B is too weak.

**Conclusion:** going smaller solves the hardware constraint but breaks the quality
requirement. qwen2.5:**3b** remains the sweet spot — it fits a 4 GB GPU and is the only
candidate that passes strict JSON at all in the full-response audit, though Stage 2 still
needs parse validation and retry/repair. 0.5B/sub-1B is not viable for this use case.
