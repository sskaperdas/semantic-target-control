$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

New-Item -ItemType Directory -Force outputs\stc_upgrade_corrected\exp16_energy_ablation | Out-Null
New-Item -ItemType Directory -Force logs\stc_corrected | Out-Null

$variants = @(
  @{
    name = "domain_range_u1"
    flags = @("--use-domain", "--use-range", "--unknown-penalty", "1.0")
  },
  @{
    name = "domain_only_u1"
    flags = @("--use-domain", "--unknown-penalty", "1.0")
  },
  @{
    name = "range_only_u1"
    flags = @("--use-range", "--unknown-penalty", "1.0")
  },
  @{
    name = "domain_range_u0p5"
    flags = @("--use-domain", "--use-range", "--unknown-penalty", "0.5")
  },
  @{
    name = "domain_range_u2"
    flags = @("--use-domain", "--use-range", "--unknown-penalty", "2.0")
  }
)

$jobs = @(
  @{
    key = "eurostat_pairre"
    datasetName = "EurostatKG_PairRE"
    processedDir = "data\processed\eurostatkg"
    runDir = "eurostat_schema_portfolio_v1\best_model"
    topM = "20000"
    batch = "32"
    logEvery = "500"
    extra = @()
  },
  @{
    key = "dbpedia_complex"
    datasetName = "DBpedia_ComplEx"
    processedDir = "data\processed\dbpedia"
    runDir = "dbpedia_schema_portfolio_safe_v3\best_model"
    topM = "5000"
    batch = "8"
    logEvery = "500"
    extra = @("--query-id-file", "outputs\stc_upgrade\audit\dbpedia_paper_subset_query_ids.txt")
  }
)

foreach ($job in $jobs) {
  foreach ($variant in $variants) {
    $outDir = "outputs\stc_upgrade_corrected\exp16_energy_ablation\$($job.key)\$($variant.name)"
    $logPath = "logs\stc_corrected\exp16_$($job.key)_$($variant.name).txt"

    Write-Host ""
    Write-Host "================================================================================"
    Write-Host "Running EXP16 $($job.key) / $($variant.name)"
    Write-Host "Output: $outDir"
    Write-Host "Log: $logPath"
    Write-Host "================================================================================"

    $argsList = @(
      "scripts\run_exp14_factual_utility.py",
      "--processed-dir", $job.processedDir,
      "--run-dir", $job.runDir,
      "--output-dir", $outDir,
      "--dataset-name", $job.datasetName,
      "--split", "test",
      "--mode", "all",
      "--top-k", "10",
      "--top-m", $job.topM,
      "--quota", "5",
      "--query-batch-size", $job.batch,
      "--device", "cuda",
      "--seed", "42",
      "--check-policy", "available_any",
      "--summary-scopes", "full", "blind_strict",
      "--log-every", $job.logEvery
    ) + $job.extra + $variant.flags

    & python @argsList 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
      throw "Python run failed with exit code $LASTEXITCODE for $($job.key) / $($variant.name). See $logPath"
    }
  }
}

Write-Host ""
Write-Host "EXP16 energy ablation runs finished."
