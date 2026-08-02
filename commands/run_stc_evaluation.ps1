[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("08", "12", "14", "19", "21", "22")]
    [string]$Experiment,

    [Parameter(Mandatory = $true)]
    [string]$ProcessedDir,

    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [Parameter(Mandatory = $true)]
    [string]$DatasetName,

    [int]$TopK = 10,
    [int]$TopM = 5000,
    [int]$Quota = 5,
    [int[]]$Quotas = @(1, 3, 5, 10),
    [int]$QueryBatchSize = 32,
    [string]$Device = "cuda",
    [string]$CheckPolicy = "available_any",
    [string]$QueryIdFile,
    [double]$UnknownPenalty = 1.0,
    [switch]$UseDisjoint,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$Scripts = @{
    "08" = "run_exp08_global_quota_return_semantics.py"
    "12" = "run_exp12_quota_matched_baselines.py"
    "14" = "run_exp14_factual_utility.py"
    "19" = "run_exp19_bootstrap_ci.py"
    "21" = "run_exp21_runtime_breakdown.py"
    "22" = "run_exp22_learned_reranker_baseline.py"
}

$ScriptPath = Join-Path $Root ("scripts\" + $Scripts[$Experiment])
if (-not (Test-Path -LiteralPath $ScriptPath)) {
    throw "Experiment script not found: $ScriptPath"
}

$Arguments = @(
    "--processed-dir", $ProcessedDir,
    "--run-dir", $RunDir,
    "--output-dir", $OutputDir,
    "--dataset-name", $DatasetName,
    "--mode", "all",
    "--top-k", $TopK,
    "--top-m", $TopM,
    "--query-batch-size", $QueryBatchSize,
    "--device", $Device,
    "--seed", "42",
    "--check-policy", $CheckPolicy,
    "--use-domain",
    "--use-range",
    "--unknown-penalty", $UnknownPenalty
)

if ($UseDisjoint) {
    $Arguments += "--use-disjoint"
}
if (-not [string]::IsNullOrWhiteSpace($QueryIdFile)) {
    $Arguments += @("--query-id-file", $QueryIdFile)
}

switch ($Experiment) {
    "08" {
        $Arguments += @("--split", "test", "--quotas")
        $Arguments += $Quotas
    }
    "12" {
        $Arguments += @(
            "--split", "test",
            "--quotas"
        )
        $Arguments += $Quotas
        $Arguments += @(
            "--binary-like",
            "--query-scope", "full"
        )
    }
    "14" {
        $Arguments += @(
            "--split", "test",
            "--quota", $Quota,
            "--summary-scopes", "full", "blind_strict",
            "--binary-like"
        )
    }
    "19" {
        $Arguments += @(
            "--split", "test",
            "--quota", $Quota,
            "--summary-scopes", "full", "blind_strict",
            "--binary-like",
            "--bootstrap-samples", "2000",
            "--bootstrap-seed", "42"
        )
    }
    "21" {
        $Arguments += @(
            "--split", "test",
            "--quota", $Quota,
            "--binary-like"
        )
    }
    "22" {
        $Arguments += @(
            "--train-split", "train",
            "--valid-split", "valid",
            "--eval-split", "test",
            "--quota", $Quota,
            "--binary-like",
            "--summary-scopes", "full", "blind_strict",
            "--learner", "lightgbm",
            "--label-policy", "admissible",
            "--alphas", "0", "0.03", "0.1", "0.3", "1", "3", "10", "30",
            "--selection-scope", "blind_strict",
            "--selection-quota-success-threshold", "0.95"
        )
    }
}

& $Python $ScriptPath @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Experiment $Experiment failed."
}
