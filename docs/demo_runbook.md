# Presentation demo runbook — fine-tuning, without the wait

**Golden rule: never train to completion live.** A full fine-tune is ~hours (≈5 h on a
16 GB laptop). The demo shows the *finished* model instantly, and — optionally — a
2-minute mini-train so the audience sees real learning happen. Everything below uses
artifacts that already exist.

## The 4-minute demo flow

**1. One data contract (30 s).** Show one training example — a `messages` array with the
Databricks study-note system prompt, a source chunk, and the target notes:
```bash
head -1 data/processed/training/databricks_ld_foundations/train.jsonl | python3 -m json.tool
```
Say: "Both the Mac and HPC backends train on exactly this."

**2. Watch it learn — 2-3 min mini-train (optional but great).** Real training, real loss
drop, tiny so it finishes on stage:
```bash
scripts/demo_train.sh
```
Point at the falling `Val loss` (≈1.1 → ≈0.7). Say: "This is the real thing, just 40
iterations. The full run does ~600 — which we're **not** waiting for; here's the result."

**3. The result — instant, pre-trained (the star).** The already-fine-tuned model runs
immediately:
```bash
ollama run edge-slm-databricks "Turn this into structured study notes: <paste any Databricks content>"
```
It produces the structured study-note schema it learned. This is the payoff — no waiting.

**4. Why 600, not 1400 (10 s of credibility).** Show `docs/mlx_loss_curve.svg`: validation
loss bottoms at ~iter 600 then overfits. "We caught the model over-training and picked the
right checkpoint" — shows real ML rigor.

## Cross-platform story (say it, show one)

Same command, three targets (docs/finetuning_pipeline.md):
```bash
PYTHONPATH=pipeline python -m training.train --data-dir ... --output-dir ...
# prints: "Apple Silicon detected -> MLX (Metal) QLoRA" and runs
```
Mac is live; Windows/NVIDIA and Pawsey/HPC are the same pipeline (a teammate runs those
on their hardware).

## Pre-flight checklist (do this BEFORE the talk)

- [ ] `ollama run edge-slm-databricks "hi"` returns instantly (model loaded, warm).
- [ ] `scripts/demo_train.sh` completes and shows a loss drop (run it once to warm caches).
- [ ] The base model is cached (no download mid-demo): it is, if the mini-train ran.
- [ ] Have `docs/mlx_loss_curve.svg` open in a tab.

## Fallbacks (things that go wrong live)

- **Wi-Fi dies / model tries to download** → everything is cached locally; the mini-train
  and `ollama run` need no network. (Don't `ollama pull` anything live.)
- **Mini-train runs long / laptop is hot** → skip step 2, go straight to `ollama run`. The
  loss curve slide tells the training story without a live run.
- **Ollama not running** → `ollama serve &` then retry; the model is `edge-slm-databricks`.
- **Someone asks "can we retrain now?"** → "A full run is ~hours; that's why we ship the
  trained adapter and the ready model" (GitHub Release + the share bundle).
