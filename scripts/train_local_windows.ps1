<#
Local Trainer Backend: QLoRA fine-tune of the Student on a Windows NVIDIA laptop.

Consumes train.jsonl / valid.jsonl (chat format) and emits a LoRA adapter — the
Windows counterpart of scripts/train_local_mlx.sh. Local runs are development
artifacts; the Pawsey run is canonical (ADR 0003).

Requires:  pip install -r requirements-training-windows.txt   (+ CUDA torch)

Usage:
  .\scripts\train_local_windows.ps1 <training_data_dir> <adapter_output_dir>

Example:
  .\scripts\train_local_windows.ps1 `
    data\processed\training\databricks_ld_foundations `
    data\processed\adapters\databricks_ld_foundations_win
#>
param(
  [Parameter(Mandatory = $true)][string]$DataDir,
  [Parameter(Mandatory = $true)][string]$OutputDir
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "pipeline"

# batch 1 + grad-accum 16 + seq 2048 keeps QLoRA of a 4B model within ~8 GB VRAM.
# On a larger card, raise --batch-size / --max-seq-length or drop --load-in-4bit.
python -m training.train_trl `
  --data-dir $DataDir `
  --output-dir $OutputDir `
  --load-in-4bit `
  --batch-size 1 `
  --grad-accum 16 `
  --max-seq-length 2048

Write-Host "Adapter written to $OutputDir"
Write-Host "Next: merge the adapter and convert to GGUF for Ollama (see docs/finetuning_pipeline.md)."
