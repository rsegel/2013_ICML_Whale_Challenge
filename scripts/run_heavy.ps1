<#
Runs the long CPU training jobs one after another, outside any editor or agent session.

    powershell -ExecutionPolicy Bypass -File scripts\run_heavy.ps1            # everything
    powershell -ExecutionPolicy Bypass -File scripts\run_heavy.ps1 -CvOnly    # only the cross-validation folds

Safe to stop (Ctrl+C) and safe to run again: finished jobs are skipped and an
interrupted job continues from its last finished epoch. Jobs run at below-normal
priority, so interactive work on the same machine stays responsive.
#>
param(
    [switch]$CvOnly,
    [int]$Threads = 10
)

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
if (-not $env:UV_PROJECT_ENVIRONMENT) { $env:UV_PROJECT_ENVIRONMENT = "$HOME\dv_cache\venv" }
$cache = if ($env:WHALE_CACHE_DIR) { $env:WHALE_CACHE_DIR } else { "$HOME\dv_cache" }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
(Get-Process -Id $PID).PriorityClass = "BelowNormal"   # child processes inherit this

# name = the predictions file the job writes; args = what to pass to whale.train
$jobs = @(
    @{ name = "resnet18_finetune_fold2_seed0"; args = "--config configs/baseline.yaml --fold 2" },
    @{ name = "resnet18_finetune_fold3_seed0"; args = "--config configs/baseline.yaml --fold 3" }
)
if (-not $CvOnly) {
    $jobs += @(
        # label efficiency on the design fold: 10% and 1% of the training clips
        @{ name = "resnet18_finetune_fold0_seed0_frac0.1";  args = "--config configs/baseline.yaml --fold 0 --train-fraction 0.1" },
        @{ name = "resnet18_finetune_fold0_seed0_frac0.01"; args = "--config configs/baseline.yaml --fold 0 --train-fraction 0.01" },
        # frozen backbone, linear head only
        @{ name = "resnet18_frozen_fold0_seed0"; args = "--config configs/baseline_frozen.yaml --fold 0" }
    )
}

foreach ($job in $jobs) {
    $name = $job.name
    if (Test-Path "$cache\predictions\$name.parquet") {
        Write-Host "[skip] $name (predictions exist)"
        continue
    }
    $log = "$cache\$name.log"
    Write-Host "`n[start $(Get-Date -Format HH:mm)] $name   log: $log"
    cmd /c "uv run python -m whale.train $($job.args) --threads $Threads 2>&1" | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[failed $(Get-Date -Format HH:mm)] $name (exit $LASTEXITCODE). Run this script again to resume."
        exit $LASTEXITCODE
    }
    Write-Host "[done $(Get-Date -Format HH:mm)] $name"
}
Write-Host "`nAll jobs finished at $(Get-Date -Format HH:mm)."
