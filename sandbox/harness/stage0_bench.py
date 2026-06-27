#!/usr/bin/env python3
"""
Stage 0 benchmark harness — Edge SLM pipeline.

Measures each candidate model in ISOLATION (one resident at a time) against a
fixed prompt set, recording memory footprint, peak RAM/swap, tokens/sec, and —
when a real GPU is present — VRAM usage.

Methodology mirrors the team's existing Stage 0 report (Kirill's isolated-baseline
harness) so results stay comparable:
  1. Evict every resident model and wait until the runtime reports nothing loaded.
  2. Sample a clean baseline.
  3. Load the model, measure the after-load footprint (delta from baseline).
  4. Run the prompt set while a background sampler polls memory at 10 Hz.
  5. Record peak RAM / peak swap, and tokens/sec from the runtime's own stats.

Memory is read cgroup-aware: inside a memory-capped container the cgroup limit and
usage are used (psutil's /proc/meminfo reflects the host, not the cap). On a bare
host / VM it falls back to psutil.

Usage:
    python stage0_bench.py --label clean
    python stage0_bench.py --label multitask --models qwen2.5:3b,gemma3:4b
    OLLAMA_HOST=http://ollama:11434 python stage0_bench.py --label clean

VRAM-budget mode (faithful, preferred over the iogpu.wired_limit_mb cap): simulate a
GPU VRAM budget by setting Ollama num_gpu so surplus layers offload to CPU like a real
VRAM-limited card -- yields slow-but-working output instead of empty failures:
    python stage0_bench.py --label budget4 --vram-budget-gb 4

Quality-audit mode stores full responses and deterministic parser-oriented quality
metrics for each fixed prompt:
    python stage0_bench.py --label quality-clean --save-responses
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
SAMPLE_HZ = 10
SETTLE_SECONDS = 2.0
DEFAULT_MODELS = ["qwen2.5:3b", "gemma3:4b", "llama3.2:3b", "phi3.5"]


# --------------------------------------------------------------------------- #
# Memory readers (cgroup-aware)
# --------------------------------------------------------------------------- #
def _read_int(path: str) -> int | None:
    try:
        with open(path) as fh:
            val = fh.read().strip()
        return None if val in ("max", "") else int(val)
    except (OSError, ValueError):
        return None


def _macos_mem_used_bytes() -> int:
    """Used memory in bytes on macOS via `vm_stat` (active+wired+compressed).

    psutil.virtual_memory() uses host_statistics64 directly and throws
    intermittently on recent macOS kernels ('array not large enough'); the
    vm_stat CLI handles the buffer correctly. Never raises.
    """
    try:
        out = subprocess.check_output(["vm_stat"], text=True, timeout=5)
        import re
        mp = re.search(r"page size of (\d+) bytes", out)
        page = int(mp.group(1)) if mp else 4096

        def pages(label: str) -> int:
            m = re.search(rf"{label}:\s+(\d+)\.", out)
            return int(m.group(1)) if m else 0

        used_pages = (pages("Pages active") + pages("Pages wired down")
                      + pages("Pages occupied by compressor"))
        return used_pages * page
    except Exception:
        try:
            import psutil
            return psutil.virtual_memory().used
        except Exception:
            return 0


def _host_swap_used_bytes() -> int:
    """Swap used in bytes. psutil.swap_memory() throws intermittently on recent
    macOS (host_statistics64 HOST_VM_INFO64 'array not large enough'), so on
    Darwin read `sysctl vm.swapusage` directly; psutil elsewhere. Never raises.
    """
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "vm.swapusage"], text=True, timeout=5)
            # "total = 6144.00M  used = 4657.19M  free = 1486.81M  (encrypted)"
            import re
            m = re.search(r"used\s*=\s*([\d.]+)([KMGT])", out)
            if m:
                scale = {"K": 1024, "M": 1024 ** 2,
                         "G": 1024 ** 3, "T": 1024 ** 4}[m.group(2)]
                return int(float(m.group(1)) * scale)
        except Exception:
            return 0
        return 0
    try:
        import psutil
        return psutil.swap_memory().used
    except Exception:
        return 0


class MemorySource:
    """Reads 'used' and 'swap' bytes from cgroup v2/v1 if capped, else psutil."""

    def __init__(self) -> None:
        self.kind = "psutil"
        self.limit_bytes: int | None = None
        # cgroup v2
        if os.path.exists("/sys/fs/cgroup/memory.current"):
            self.kind = "cgroup-v2"
            self.limit_bytes = _read_int("/sys/fs/cgroup/memory.max")
        # cgroup v1
        elif os.path.exists("/sys/fs/cgroup/memory/memory.usage_in_bytes"):
            self.kind = "cgroup-v1"
            lim = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            # v1 reports a huge sentinel when unlimited
            self.limit_bytes = lim if lim and lim < (1 << 62) else None

    def sample(self) -> tuple[int, int]:
        """Return (used_bytes, swap_bytes)."""
        if self.kind == "cgroup-v2":
            used = _read_int("/sys/fs/cgroup/memory.current") or 0
            swap = _read_int("/sys/fs/cgroup/memory.swap.current") or 0
            return used, swap
        if self.kind == "cgroup-v1":
            used = _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes") or 0
            swap_total = _read_int(
                "/sys/fs/cgroup/memory/memory.memsw.usage_in_bytes"
            )
            swap = max(0, (swap_total or used) - used)
            return used, swap
        if sys.platform == "darwin":
            return _macos_mem_used_bytes(), _host_swap_used_bytes()
        import psutil  # local import: not needed in container path
        return psutil.virtual_memory().used, _host_swap_used_bytes()

    def total_bytes(self) -> int | None:
        if self.limit_bytes:
            return self.limit_bytes
        if sys.platform == "darwin":
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True, timeout=5)
                return int(out.strip())
            except Exception:
                return None
        try:
            import psutil

            return psutil.virtual_memory().total
        except Exception:
            return None


def _nvidia_vram_used_mb() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return max(int(line) for line in out.strip().splitlines())
    except Exception:
        return None


def _amd_vram_used_mb() -> int | None:
    if not shutil.which("rocm-smi"):
        return None
    try:
        # rocm-smi --showmeminfo vram --json -> {"card0": {"VRAM Total Used Memory (B)": "..."}}
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            text=True, timeout=5,
        )
        data = json.loads(out)
        used = []
        for card in data.values():
            for k, v in card.items():
                if "used" in k.lower() and "vram" in k.lower():
                    used.append(int(v) // (1024 * 1024))
        return max(used) if used else None
    except Exception:
        return None


def _apple_gpu_used_mb() -> int | None:
    """Apple-Silicon GPU memory in use (MB), read from the IOAccelerator stats.

    On unified-memory Macs the GPU has no separate pool, but the driver still
    reports how much system memory is currently wired for the GPU. We read
    PerformanceStatistics."In use system memory" via ioreg. This is system-wide
    GPU usage; the harness subtracts a per-model baseline to isolate the model.
    """
    if sys.platform != "darwin" or not shutil.which("ioreg"):
        return None
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            text=True, timeout=5,
        )
        import re
        vals = re.findall(r'"In use system memory"=(\d+)', out)
        if not vals:
            return None
        return max(int(v) for v in vals) // (1024 * 1024)
    except Exception:
        return None


def gpu_vram_used_mb() -> int | None:
    """GPU memory used (MB): NVIDIA (nvidia-smi), AMD (rocm-smi), or Apple (ioreg).

    Apple Silicon HAS measurable GPU memory — it is a wired slice of unified
    memory, reported here. It can also be CAPPED via iogpu.wired_limit_mb to
    simulate a discrete-VRAM budget (see apple_gpu_wired_limit_mb).
    """
    return _nvidia_vram_used_mb() or _amd_vram_used_mb() or _apple_gpu_used_mb()


def apple_gpu_wired_limit_mb() -> int | None:
    """Current GPU wired-memory cap (MB). 0 = macOS default (~75% of RAM).

    Set non-zero (sudo sysctl iogpu.wired_limit_mb=8192) to simulate an 8 GB
    VRAM ceiling. Recorded in the report so a capped run is self-documenting.
    """
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "iogpu.wired_limit_mb"], text=True, timeout=5)
        return int(out.strip())
    except Exception:
        return None


def detect_accelerator() -> str:
    """Best-effort label of the active accelerator for the report header."""
    if shutil.which("nvidia-smi"):
        return "nvidia-cuda"
    if shutil.which("rocm-smi"):
        return "amd-rocm"
    if sys.platform == "darwin":
        return "apple-metal"
    return "cpu"


# --------------------------------------------------------------------------- #
# Background memory sampler
# --------------------------------------------------------------------------- #
class PeakSampler(threading.Thread):
    def __init__(self, src: MemorySource) -> None:
        super().__init__(daemon=True)
        self.src = src
        self._stop_event = threading.Event()
        self.peak_used = 0
        self.peak_swap = 0
        self.peak_vram_mb = 0
        self.samples = 0
        self.errors = 0
        self.last_error = None

    def run(self) -> None:
        interval = 1.0 / SAMPLE_HZ
        # Memory (cheap, cgroup/psutil read) at full rate; GPU (spawns a
        # subprocess: ioreg/nvidia-smi/rocm-smi) every 5th tick (~2 Hz) so it
        # doesn't perturb the tokens/sec measurement.
        gpu_every = max(1, SAMPLE_HZ // 2)
        while not self._stop_event.is_set():
            try:
                used, swap = self.src.sample()
                self.peak_used = max(self.peak_used, used)
                self.peak_swap = max(self.peak_swap, swap)
                if self.samples % gpu_every == 0:
                    vram = gpu_vram_used_mb()
                    if vram is not None:
                        self.peak_vram_mb = max(self.peak_vram_mb, vram)
            except Exception as e:  # a transient read must never kill the sampler
                self.errors += 1
                self.last_error = str(e)
            self.samples += 1
            time.sleep(interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)


# --------------------------------------------------------------------------- #
# Ollama control
# --------------------------------------------------------------------------- #
def ollama_loaded() -> list[dict]:
    r = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def evict_all(timeout: float = 60.0) -> None:
    """Unload every resident model (keep_alive=0) and wait for an empty runtime."""
    for m in ollama_loaded():
        try:
            requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": m["name"], "keep_alive": 0},
                timeout=30,
            )
        except requests.RequestException:
            pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not ollama_loaded():
            return
        time.sleep(1)
    raise RuntimeError("runtime still has models resident after evict timeout")


def generate(model: str, prompt: str, num_gpu: int | None = None) -> dict:
    options = {"temperature": 0.1, "num_predict": 256}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu  # layers to offload to GPU (None = Ollama default)
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",
            "options": options,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def score_response(prompt_id: str, response: str) -> dict:
    """Deterministic quality checks for the fixed Stage 0 prompt set.

    These are intentionally narrow parser/format checks, not subjective grading.
    They let the report distinguish "model ran" from "model obeyed the output
    contract needed downstream", especially for strict JSON generation.
    """
    stripped = response.strip()
    lower = stripped.lower()
    starts_fence = stripped.startswith("```")
    contains_cjk = bool(re.search(r"[\u4e00-\u9fff]", stripped))
    metrics = {
        "nonempty": bool(stripped),
        "response_chars": len(response),
        "starts_markdown_fence": starts_fence,
        "contains_cjk": contains_cjk,
    }

    if prompt_id == "bullet_count_compliance":
        lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
        bullet_lines = [ln for ln in lines if ln.startswith("- ")]
        metrics.update({
            "nonempty_line_count": len(lines),
            "bullet_line_count": len(bullet_lines),
            "exactly_five_dash_bullets": len(lines) == 5 and len(bullet_lines) == 5,
        })
        metrics["pass"] = metrics["exactly_five_dash_bullets"]

    elif prompt_id == "grounded_answering":
        has_quant = "gguf" in lower and "q4_k_m" in lower
        denies_interactive = any(s in lower for s in [
            "no interactive", "no one", "does not run", "do not run",
            "no interactive access", "not interactively", "no access",
        ])
        metrics.update({
            "mentions_gguf_q4km": has_quant,
            "denies_interactive_access": denies_interactive,
        })
        metrics["pass"] = has_quant and denies_interactive

    elif prompt_id == "strict_json_pairs":
        json_ok = False
        item_count = None
        exact_schema = False
        parse_error = None
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                item_count = len(parsed)
                exact_schema = (
                    item_count == 2
                    and all(isinstance(x, dict) for x in parsed)
                    and all(set(x.keys()) == {"instruction", "response"} for x in parsed)
                    and all(isinstance(x["instruction"], str)
                            and isinstance(x["response"], str) for x in parsed)
                )
                json_ok = exact_schema
        except Exception as e:
            parse_error = f"{type(e).__name__}: {e}"
        metrics.update({
            "strict_json_parse_ok": json_ok,
            "strict_json_item_count": item_count,
            "strict_json_exact_schema": exact_schema,
            "strict_json_parse_error": parse_error,
        })
        metrics["pass"] = json_ok and not starts_fence and not contains_cjk

    elif prompt_id == "policy_vs_procedure":
        sentences = [s for s in re.split(r"[.!?]+", stripped) if s.strip()]
        has_policy = "policy" in lower or "policies" in lower
        has_procedure = "procedure" in lower or "procedures" in lower
        metrics.update({
            "sentence_count": len(sentences),
            "mentions_policy": has_policy,
            "mentions_procedure": has_procedure,
            "within_3_4_sentences": 3 <= len(sentences) <= 4,
        })
        metrics["pass"] = has_policy and has_procedure and metrics["within_3_4_sentences"]

    elif prompt_id == "honest_unknown":
        unknown_phrases = [
            "do not have", "don't have", "not have", "do not know", "don't know",
            "unknown", "not available", "not publicly", "not stated",
            "cannot provide", "no access", "unable to",
        ]
        has_unknown = any(s in lower for s in unknown_phrases)
        has_dollar_figure = bool(re.search(
            r"(\$\s*\d|usd\s*\d|\b\d[\d,]*(?:\.\d+)?\s*(?:dollars|aud|usd)\b)",
            lower,
        ))
        metrics.update({
            "admits_unknown": has_unknown,
            "contains_dollar_figure": has_dollar_figure,
        })
        metrics["pass"] = has_unknown and not has_dollar_figure

    else:
        metrics["pass"] = bool(stripped)

    return metrics


def model_block_count(model: str) -> int | None:
    """Total transformer layers for a model, from /api/show (<arch>.block_count)."""
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/show",
                          json={"model": model}, timeout=30)
        r.raise_for_status()
        info = r.json().get("model_info", {})
        for k, v in info.items():
            if k.endswith(".block_count"):
                return int(v)
    except Exception:
        pass
    return None


def ps_size_for(model: str) -> dict:
    for m in ollama_loaded():
        if m["name"] == model or m["name"].split(":")[0] == model.split(":")[0]:
            return {"size_bytes": m.get("size"), "size_vram_bytes": m.get("size_vram")}
    return {"size_bytes": None, "size_vram_bytes": None}


# --------------------------------------------------------------------------- #
# Benchmark one model
# --------------------------------------------------------------------------- #
GB = 1024 ** 3


def bench_model(model: str, prompts: list[dict], src: MemorySource,
                vram_budget_gb: float | None = None,
                save_responses: bool = False) -> dict:
    print(f"\n=== {model} ===", flush=True)

    print("  evicting resident models...", flush=True)
    evict_all()
    time.sleep(SETTLE_SECONDS)
    base_used, base_swap = src.sample()
    base_vram = gpu_vram_used_mb()
    print(f"  baseline: used={base_used/GB:.2f} GB swap={base_swap/GB:.2f} GB "
          f"vram={base_vram if base_vram is not None else 'n/a'} MB", flush=True)

    print("  loading (warm-up generate)...", flush=True)
    warm = generate(model, "Reply with the single word: ready.")
    time.sleep(SETTLE_SECONDS)
    load_used, load_swap = src.sample()
    load_vram = gpu_vram_used_mb()
    sizes = ps_size_for(model)

    # VRAM-budget mode: instead of capping GPU wired memory (which makes Ollama
    # over-commit and emit empty output), set num_gpu so Ollama proactively offloads
    # layers to CPU like a real VRAM-limited GPU would. We measured the full GPU
    # footprint above; map the budget to a layer count and reload constrained.
    num_gpu = None
    budget_info = None
    if vram_budget_gb is not None:
        L = model_block_count(model)
        full_vram_b = sizes.get("size_vram_bytes")
        if L and full_vram_b:
            per_layer = full_vram_b / L
            num_gpu = max(0, min(L, round((vram_budget_gb * GB) / per_layer)))
            print(f"  budget {vram_budget_gb} GB: layers={L} full_vram="
                  f"{full_vram_b/GB:.2f} GB -> num_gpu={num_gpu}/{L}", flush=True)
            evict_all()
            time.sleep(SETTLE_SECONDS)
            generate(model, "Reply with the single word: ready.", num_gpu=num_gpu)
            time.sleep(SETTLE_SECONDS)
            load_used, load_swap = src.sample()
            load_vram = gpu_vram_used_mb()
            sizes = ps_size_for(model)  # achieved footprint under the constraint
            budget_info = {
                "vram_budget_gb": vram_budget_gb,
                "block_count": L,
                "num_gpu": num_gpu,
                "full_vram_gb": round(full_vram_b / GB, 2),
                "achieved_vram_gb": (round(sizes["size_vram_bytes"] / GB, 2)
                                     if sizes.get("size_vram_bytes") else None),
            }
        else:
            print("  budget mode: no block_count/full_vram; running unconstrained",
                  flush=True)

    sampler = PeakSampler(src)
    sampler.start()

    tok_per_s: list[float] = []
    prompt_records = []
    for p in prompts:
        res = generate(model, p["prompt"], num_gpu=num_gpu)
        response = res.get("response", "")
        ec = res.get("eval_count")
        ed = res.get("eval_duration")  # nanoseconds
        tps = (ec / (ed / 1e9)) if ec and ed else None
        if tps:
            tok_per_s.append(tps)
        record = {
            "id": p["id"],
            "eval_count": ec,
            "tokens_per_sec": round(tps, 1) if tps else None,
            "response_preview": response[:200],
            "quality": score_response(p["id"], response),
        }
        if save_responses:
            record["response"] = response
        prompt_records.append(record)
        print(f"  prompt {p['id']}: {round(tps,1) if tps else 'n/a'} tok/s", flush=True)

    sampler.stop()

    quality_passes = sum(1 for p in prompt_records if p["quality"].get("pass"))
    nonempty = sum(1 for p in prompt_records if p["quality"].get("nonempty"))
    strict = next((p["quality"] for p in prompt_records
                   if p["id"] == "strict_json_pairs"), {})
    bullets = next((p["quality"] for p in prompt_records
                    if p["id"] == "bullet_count_compliance"), {})

    def gb(x: int) -> float:
        return round(x / GB, 2)

    return {
        "model": model,
        "memory_source": src.kind,
        "vram_budget": budget_info,
        "baseline_used_gb": gb(base_used),
        "load_footprint_gb": gb(max(0, load_used - base_used)),
        "peak_used_gb": gb(sampler.peak_used),
        "peak_swap_gb": gb(sampler.peak_swap),
        "load_swap_gb": gb(load_swap),
        "ollama_size_gb": gb(sizes["size_bytes"]) if sizes["size_bytes"] else None,
        "ollama_size_vram_gb": (
            gb(sizes["size_vram_bytes"]) if sizes["size_vram_bytes"] else None
        ),
        "baseline_vram_used_mb": base_vram,
        "load_vram_used_mb": load_vram,
        "peak_vram_used_mb": sampler.peak_vram_mb or None,
        "mean_tokens_per_sec": round(statistics.mean(tok_per_s), 1) if tok_per_s else None,
        "quality_summary": {
            "passes": quality_passes,
            "total": len(prompt_records),
            "nonempty": nonempty,
            "strict_json_pass": bool(strict.get("pass")),
            "strict_json_parse_ok": bool(strict.get("strict_json_parse_ok")),
            "strict_json_fence": bool(strict.get("starts_markdown_fence")),
            "strict_json_contains_cjk": bool(strict.get("contains_cjk")),
            "exactly_five_bullets": bool(bullets.get("exactly_five_dash_bullets")),
        },
        "prompts": prompt_records,
        "warmup_response": warm.get("response", "")[:120],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_prompts(prompts_dir: Path) -> list[dict]:
    f = prompts_dir / "stage0_prompts.json"
    if not f.exists():
        sys.exit(f"prompt set not found: {f}")
    return json.loads(f.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0 SLM benchmark harness")
    ap.add_argument("--label", required=True,
                    help="condition label, e.g. clean / multitask")
    ap.add_argument("--models", default=os.environ.get("MODELS", ""),
                    help="comma-separated model tags (default: the 4 candidates)")
    ap.add_argument("--out", default="results", help="output directory root")
    ap.add_argument("--prompts-dir", default=str(Path(__file__).parent / "prompts"))
    ap.add_argument("--vram-budget-gb", type=float, default=None,
                    help="simulate a VRAM budget by setting Ollama num_gpu (proactive "
                         "layer offload), instead of capping wired memory")
    ap.add_argument("--save-responses", action="store_true",
                    help="store full model responses in results.json for quality audit")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()] or DEFAULT_MODELS
    prompts = load_prompts(Path(args.prompts_dir))
    src = MemorySource()

    # connectivity check
    try:
        ollama_loaded()
    except requests.RequestException as e:
        sys.exit(f"cannot reach Ollama at {OLLAMA_HOST}: {e}")

    cap = src.total_bytes()
    cap_str = f"cap={cap/GB:.1f} GB" if cap else "cap=none"
    print(f"runtime={OLLAMA_HOST}  accel={detect_accelerator()}  "
          f"mem_source={src.kind}  {cap_str}", flush=True)

    if args.vram_budget_gb is not None:
        print(f"VRAM-budget mode: {args.vram_budget_gb} GB (num_gpu offload)", flush=True)

    results = []
    for model in models:
        try:
            results.append(bench_model(model, prompts, src,
                                       vram_budget_gb=args.vram_budget_gb,
                                       save_responses=args.save_responses))
        except Exception as e:  # fail loud, keep going
            import traceback
            tb = traceback.format_exc()
            print(f"  !! {model} FAILED: {e}\n{tb}", flush=True)
            results.append({"model": model, "error": str(e), "traceback": tb})

    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ollama_host": OLLAMA_HOST,
        "memory_source": src.kind,
        "mem_cap_gb": round(cap / GB, 1) if cap else None,
        "accelerator": detect_accelerator(),
        "gpu_present": gpu_vram_used_mb() is not None,
        "apple_gpu_wired_limit_mb": apple_gpu_wired_limit_mb(),
        "full_responses_saved": args.save_responses,
        "models": results,
    }
    out_file = out_dir / "results.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
