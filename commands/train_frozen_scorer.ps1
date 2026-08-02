[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("pykeen", "ddp", "schema-portfolio")]
    [string]$Trainer,

    [Parameter(Mandatory = $true)]
    [string]$ProcessedDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [string]$DatasetName,
    [string]$Model = "complex",
    [string]$Portfolio = "medium",
    [string]$Devices = "cuda:0",
    [int]$Seed = 42,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

switch ($Trainer) {
    "pykeen" {
        & $Python `
            "$Root\scripts\training\train_base_kgc.py" `
            --processed-dir $ProcessedDir `
            --output-dir $OutputDir `
            --dataset-name $DatasetName `
            --portfolio $Portfolio `
            --seed $Seed `
            --device $Devices `
            --filtered-eval
    }

    "schema-portfolio" {
        & $Python `
            "$Root\scripts\training\train_schema_aware_portfolio_multigpu.py" `
            --processed-dir $ProcessedDir `
            --output-dir $OutputDir `
            --dataset-name $DatasetName `
            --portfolio $Portfolio `
            --devices $Devices `
            --seed $Seed `
            --filtered-eval `
            --checkpoint-on-failure
    }

    "ddp" {
        Write-Warning "Launch this wrapper through torchrun for true multi-process DDP."
        & $Python `
            "$Root\scripts\training\train_base_kgc_ddp.py" `
            --processed-dir $ProcessedDir `
            --output-dir $OutputDir `
            --model $Model `
            --seed $Seed `
            --dump-topm `
            --clear-cuda-cache `
            --alloc-expandable-segments
    }
}

if ($LASTEXITCODE -ne 0) {
    throw "Frozen-scorer training failed."
}
