# Edge SLM — Sandbox

Reproducible, resource-capped environments for **Stage 0 model/hardware validation**
and for running the downstream pipeline under controlled conditions.

The purpose of this sandbox is twofold:

1. **Reproducibility** — pin one runtime (Ollama), one harness, one prompt set, so any
   teammate produces comparable numbers. This fixes the "two reports, two different
   stacks (Ollama on macOS vs LM Studio on Windows)" inconsistency from the first round.
2. **Resource simulation** — cap memory to the project's **16 GB RAM** target and observe
   behaviour (footprint, peak RAM, swap, tokens/sec) deterministically.

## ⚠️ What this sandbox can and cannot validate

The target deployment machine is a **16 GB RAM laptop with a discrete NVIDIA GPU
(2–8 GB VRAM)**. The decisive metric in both Stage 0 reports was **VRAM headroom**.

On an Apple-Silicon host (this repo's dev machine) there is **no NVIDIA GPU**, so:

| Axis                        | Docker CPU (Apple host) | **Native Metal (Apple host)** | Docker CUDA (NVIDIA host) | Docker ROCm (AMD Linux host) | Parallels VM (Apple host) |
| --------------------------- | :---------------------: | :---------------------------: | :-----------------------: | :--------------------------: | :-----------------------: |
| Reproducible pinned env     |           ✅            |              ✅               |            ✅             |             ✅               |            ✅             |
| 16 GB system-RAM cap        |           ✅            |    ⚠️ soft¹                   |            ✅             |             ✅               |            ✅             |
| GPU acceleration            |           ❌            |       ✅ Apple GPU (Metal)    |       ✅ NVIDIA (CUDA)    |        ✅ AMD (ROCm)         |   ❌ (Apple vGPU only)    |
| **GPU memory used**         |           ❌            |    ✅ ioreg / size_vram²      |       ✅ nvidia-smi       |         ✅ rocm-smi          |            ❌             |
| **Simulated VRAM cap**      |           ❌            |  ✅ iogpu.wired_limit_mb³     |   n/a (real VRAM)         |       n/a (real VRAM)        |            ❌             |
| Transferable GPU *speed*    |           ❌            |        ❌⁴                    |            ✅             |             ✅               |            ❌             |
| Full OS / Windows behaviour |           ❌            |              ❌               |            ❌             |             ❌               |            ✅             |

¹ macOS has no cgroups — the 16 GB *system-RAM* cap can't be hard-enforced natively on a
>16 GB Mac. The harness records absolute footprint/peak/swap so you can judge 16 GB-fit;
for a hard system-RAM cap use the Parallels VM (which loses Metal).
² Apple Silicon GPU memory **is** measurable: the harness reads `IOAccelerator`'s
`"In use system memory"` via `ioreg`, plus Ollama's per-model `size_vram`.
³ Because memory is unified, `sudo sysctl iogpu.wired_limit_mb=8192` caps GPU-wired memory
to an **8 GB ceiling** — simulating the target's 2–8 GB discrete-VRAM *headroom*. Use
`make bench-metal-capped CAP_GB=8`.
⁴ Capacity ("does it fit in N GB of VRAM?") IS reproducible on Metal; *speed* (tok/s) is
not — M-series unified bandwidth and GPU cores differ from a laptop dGPU's GDDR6.

**Native Metal mode** measures this Mac's real Apple-GPU memory and can *simulate* a 2–8 GB
VRAM ceiling for the headroom question. **CUDA/ROCm modes** on real NVIDIA/AMD hardware
remain the only source of transferable VRAM *and speed* together — that's what finally
closes the "16 GB + dGPU" validation gap on the actual deployment class of machine.

Parallels on Apple Silicon virtualizes the *Apple* GPU into the guest (no NVIDIA driver,
no CUDA), so it gives a full OS + RAM cap but **not** real VRAM.

## Prerequisites & cross-platform portability

The Docker path runs the harness **inside a Linux container**, so it behaves identically
on macOS, Linux, and Windows. `make` is only a convenience wrapper — every target maps to
a plain `docker compose` command (see ["Without make"](#without-make-any-os) below), which
is what Windows users should use.

| Host OS | CPU mode | GPU mode | How to drive it |
| ------- | :------: | -------- | --------------- |
| **Linux** | ✅ | NVIDIA (CUDA) or AMD (ROCm), native | `make …` or raw `docker compose` |
| **macOS** (Apple Silicon) | ✅ (Docker) | Apple GPU only via **native Metal** (`make bench-metal`), not Docker | `make …` or raw `docker compose` |
| **Windows** | ✅ (Docker Desktop, WSL2 backend) | NVIDIA (CUDA) via **WSL2 + NVIDIA Container Toolkit** | raw `docker compose` in PowerShell, or `make` inside WSL2/Git Bash |

Requirements:
- **Docker Desktop** (macOS/Windows) or Docker Engine (Linux), with `docker compose` v2.
- ⚠️ **Docker Desktop memory must be ≥ 18 GB** for the `16g` container cap to actually bind.
  A container can't exceed the Docker VM's own RAM, so if the VM is set to 8 GB the 16 GB
  cap is silently a no-op. Set it in **Docker Desktop → Settings → Resources → Memory**.
  (Native Linux has no VM, so the cap binds directly.)
- **NVIDIA CUDA:** Linux host (or Windows WSL2) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **AMD ROCm:** **Linux-only** (amdgpu driver + ROCm). Not available on macOS/Windows.
- **Native Metal** (`make bench-metal`): macOS + Apple Silicon + `brew install ollama` only.

### Without make (any OS)

`make` is not required. The equivalent raw commands (run from `sandbox/`):

```bash
# CPU mode (any OS with Docker)
docker compose -f docker/compose.yaml -f docker/compose.cpu.yaml up -d ollama
docker exec edge-slm-ollama ollama pull qwen2.5:3b
docker compose -f docker/compose.yaml -f docker/compose.cpu.yaml run --rm bench --label clean
docker compose -f docker/compose.yaml -f docker/compose.cpu.yaml down

# CUDA mode: swap compose.cpu.yaml -> compose.cuda.yaml   (Linux / Windows-WSL2 + NVIDIA toolkit)
# ROCm mode: swap compose.cpu.yaml -> compose.rocm.yaml   (Linux + AMD only)
```

On **Windows PowerShell**, the commands are identical (Docker Desktop provides
`docker compose`); only shell env-var syntax differs (`$env:MODELS="qwen2.5:3b"` instead of
`MODELS=qwen2.5:3b`). The `docker/.env` file is the portable way to set these on every OS.

### What has actually been run vs provided

Honest status so the next person knows what's verified:

| Path | Status |
| ---- | ------ |
| Native Metal (macOS, M3 Max) — uncapped + 8/4/2 GB caps | ✅ **executed**, results in `results/` |
| Docker CPU mode | ✅ **executed end-to-end** (qwen, `accel=cpu`, `cgroup-v2`); see RAM caveat below |
| Docker CUDA / ROCm modes | 📦 **provided & compose-validated, not executed** (no NVIDIA/AMD hardware here) — run on target hardware |
| Parallels VM | 📄 **documented, not executed** |

> **Docker memory-measurement caveat.** In the Docker setup `ollama` and `bench`
> are *separate containers* with separate cgroups, so the harness's system-RAM
> numbers (`load_footprint_gb`, `peak_used_gb`) measure the tiny **bench sidecar**,
> not the model. In Docker mode rely on **tok/s** (from Ollama's API) and the
> **model VRAM** (`ollama_size_vram_gb`, and nvidia/rocm-smi in GPU modes) — these
> are correct. For true single-process system-RAM measurement, use **native** mode
> or the **Parallels VM**, where the harness shares the model's memory space.
> (A future option: co-locate the harness inside the ollama container, or read the
> ollama container's stats via the Docker API.)

## Layout

```
sandbox/
├── README.md                 # this file
├── Makefile                  # convenience targets (make help) — optional wrapper
├── docker/
│   ├── compose.yaml          # base: ollama + bench services, 16g cap
│   ├── compose.cpu.yaml      # CPU-only override (no GPU)
│   ├── compose.cuda.yaml     # NVIDIA override — real VRAM (Linux / WSL2)
│   ├── compose.rocm.yaml     # AMD ROCm override — real VRAM (Linux only)
│   ├── bench.Dockerfile      # thin python image running the harness
│   └── .env.example          # tunables (MEM_LIMIT, MODELS, ...) — copy to .env
├── harness/
│   ├── stage0_bench.py       # the benchmark (isolated baseline → footprint → peak)
│   ├── prompts/              # the 5-prompt Stage 0 set
│   └── requirements.txt
├── native/                   # non-Docker Apple-GPU paths (macOS only)
│   ├── run_metal_bench.sh    # native Metal benchmark
│   └── bench_metal_capped.sh # Metal benchmark under a simulated VRAM cap
├── vm/
│   ├── parallels-setup.md    # create a 16 GB-capped Linux / Windows-ARM VM
│   └── provision.sh          # run inside a Linux guest to install runtime + harness
├── charts/
│   ├── make_charts.py        # regenerate report SVGs from results JSON
│   ├── requirements.txt      # matplotlib
│   └── *.svg                 # generated figures (tracked)
└── results/                  # benchmark outputs (tracked — Stage 0 findings)
    ├── REPORT.md             # study-wide writeup: VRAM sweep, quality, charts
    └── <label>/results.json  # raw data per run (metal-clean, metal-budget*, ...)
```

Regenerate the report figures after new runs: `python charts/make_charts.py`.

## Quick start

### Docker, CPU mode (this Mac)

```bash
cd sandbox
make up-cpu                       # starts ollama (CPU), 16 GB cap
make pull MODEL=qwen2.5:3b        # pull the selected primary
make bench LABEL=clean            # run the harness, write results/clean/
make down
```

### Docker, CUDA mode (teammate's NVIDIA box / Pawsey)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
cd sandbox
make up-cuda                      # starts ollama with --gpus all
make pull MODEL=qwen2.5:3b
make bench LABEL=clean            # results now include real VRAM (nvidia-smi)
make down
```

### Native Metal (this Mac's Apple GPU)

Docker can't reach the Apple GPU, so real M-series performance runs natively:

```bash
cd sandbox
make bench-metal LABEL=clean      # needs: brew install ollama && brew services start ollama
```

### Docker, ROCm mode (AMD GPU on a Linux host)

```bash
cd sandbox
make up-rocm
make pull MODEL=qwen2.5:3b
make bench-rocm                    # results include real AMD VRAM (rocm-smi)
make down
```

### Parallels VM

See [vm/parallels-setup.md](vm/parallels-setup.md).

## Models

Stage 0 selected **`qwen2.5:3b`** (Q4_K_M) as primary. Backups: `gemma3:4b`,
`llama3.2:3b`, `phi3.5`. Set `MODELS` in `.env` to benchmark several in one run.
