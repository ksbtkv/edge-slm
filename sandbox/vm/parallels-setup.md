# Parallels VM — 16 GB-capped Edge SLM sandbox

A full-OS alternative to the Docker sandbox. Use this when you want to observe
behaviour inside a **real operating system** with a hard 16 GB cap — e.g. the
multitasking-RAM scenario from the Stage 0 reports — rather than a Linux container.

## ⚠️ GPU reality on Apple Silicon

Parallels Desktop on Apple Silicon **does not expose an NVIDIA GPU** to the guest.
It virtualizes the Apple GPU (DirectX 11 / OpenGL), so there is **no CUDA and no real
VRAM measurement**. Inference inside the VM runs on the **CPU**.

➡️ For real VRAM numbers, run the Docker **CUDA mode** on an actual NVIDIA machine
(see [../README.md](../README.md)). This VM covers OS realism + the RAM cap, not VRAM.

| What the VM gives you            | What it does not                     |
| -------------------------------- | ------------------------------------ |
| Real Linux or Windows-on-ARM OS  | NVIDIA CUDA / real discrete VRAM     |
| Hard 16 GB RAM cap               | x86 perf numbers (guest is ARM)      |
| CPU inference + tokens/sec       | Transferable GPU thermals/throughput |
| Multitasking-load realism        |                                      |

> Note: guests are **ARM64** (Apple Silicon host). Tools install fine, but absolute
> speed is not comparable to the target's x86 + dGPU laptop. Treat the VM as a
> *functional / RAM-pressure* check, not a perf benchmark.

## A. Linux guest (recommended, fastest path)

1. **Create the VM** — Parallels → *New* → *Download Ubuntu* (ARM64), or attach an
   Ubuntu 24.04 ARM64 ISO.
2. **Cap resources** — *Configure → Hardware*:
   - **Memory: 16384 MB** (the project target). Do **not** over-provision.
   - CPUs: 4 (matches a typical laptop; adjust to taste).
3. Boot, finish install, open a terminal in the guest.
4. **Provision** — copy `provision.sh` into the guest and run it:
   ```bash
   # inside the guest
   curl -fsSL -o provision.sh <raw-url-or-shared-folder>/provision.sh   # or use a Parallels shared folder
   bash provision.sh
   ```
   This installs Ollama, Python, the harness, pulls `qwen2.5:3b`, and runs a smoke test.
5. **Benchmark**:
   ```bash
   cd ~/edge-slm/harness
   python3 stage0_bench.py --label clean
   # open Firefox with several tabs + a couple of apps, then:
   python3 stage0_bench.py --label multitask
   ```
   Results land in `~/edge-slm/harness/results/<label>/results.json`.

> To enforce the 16 GB cap *and* observe swap exactly like the harness expects,
> the VM RAM setting is the cap. The harness reads the guest's `/proc/meminfo`
> (psutil path) since a bare VM has no memory cgroup limit.

## B. Windows-on-ARM guest (closest to the deployment OS)

Use this if you specifically need Windows behaviour (the deployment target is a
Windows laptop). Still CPU-only, still ARM.

1. Parallels → *New* → *Install Windows 11* (Parallels fetches the ARM build).
2. *Configure → Hardware → Memory: 16384 MB*.
3. In the guest, install **Ollama for Windows** (`https://ollama.com/download/windows`)
   and **Python 3.12**.
4. Copy the `harness/` folder in (drag-drop or a shared folder), then:
   ```powershell
   pip install -r requirements.txt
   python stage0_bench.py --label clean
   ```
5. For the multitasking scenario, open the real apps the report used
   (Edge with ~8–10 tabs, Teams, Outlook, VS Code) and re-run with `--label multitask`.

## Recording results

Keep the VM's `results/<label>/results.json` alongside the Docker and real-hardware
runs so the combined Stage 0 report can compare:
- Docker-CPU (Apple host), VM-CPU (Apple host) — RAM/CPU/repro axes
- Docker-CUDA (NVIDIA host) — the VRAM axis

Label each run with its environment in the filename when you archive it, e.g.
`results-vm-ubuntu-arm-clean.json`.
