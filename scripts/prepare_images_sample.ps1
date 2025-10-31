Param(
    [int]$TrainPerClass = 150,
    [int]$ValPerClass = 30,
    [int]$TestPerClass = 30,
    [string[]]$Classes = @('bacterial_blight','leaf_blast','sheath_blight','healthy')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Ensure-Dir($p) { if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null } }

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $root
try {
    Ensure-Dir 'data/tmp'
    Ensure-Dir 'data/images/train'
    Ensure-Dir 'data/images/val'
    Ensure-Dir 'data/images/test'

    Write-Host '== Checking Kaggle CLI =='
    if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
        Write-Error 'Kaggle CLI not found. Install with: pip install kaggle, then place kaggle.json to %USERPROFILE%\.kaggle\kaggle.json'
    }

    $kaggleDir = Join-Path $env:USERPROFILE '.kaggle'
    $kaggleJson = Join-Path $kaggleDir 'kaggle.json'
    if (-not (Test-Path $kaggleJson)) {
        Write-Error "kaggle.json not found at $kaggleJson. Create API token on Kaggle and place it there."
    }

    Write-Host '== Downloading datasets to data/tmp =='
    kaggle datasets download -d emmarex/plantdisease -p data/tmp -w
    kaggle datasets download -d abdallahalidev/rice-leaf-diseases -p data/tmp -w

    Write-Host '== Extracting archives =='
    Get-ChildItem data/tmp -Filter *.zip | ForEach-Object {
        $dest = Join-Path 'data/tmp' ($_.BaseName)
        if (-not (Test-Path $dest)) { Expand-Archive -Force $_.FullName $dest }
    }

    function Copy-Sample {
        param(
            [Parameter(Mandatory)] [string]$SourceGlob,
            [Parameter(Mandatory)] [string]$Dest,
            [Parameter(Mandatory)] [int]$Count
        )
        Ensure-Dir $Dest
        $all = Get-ChildItem -Recurse -File -Path $SourceGlob -Include *.jpg,*.jpeg,*.png -ErrorAction SilentlyContinue
        if (-not $all -or $all.Count -eq 0) { return }
        $take = [Math]::Min($Count, $all.Count)
        $files = $all | Get-Random -Count $take
        foreach ($f in $files) { Copy-Item $f.FullName -Destination $Dest -Force }
    }

    Write-Host '== Sampling classes into train/val/test =='
    foreach ($cls in $Classes) {
        $trainDest = Join-Path 'data/images/train' $cls
        $valDest = Join-Path 'data/images/val' $cls
        $testDest = Join-Path 'data/images/test' $cls

        $pv = 'data/tmp/plantdisease/**'
        $rice = 'data/tmp/rice-leaf-diseases/**'

        $patterns = @(
            "$pv*$cls*",
            "$pv*$(($cls -replace '_',' '))*",
            "$rice*$cls*",
            "$rice*$(($cls -replace '_',' '))*"
        )

        $matched = $false
        foreach ($pat in $patterns) {
            $cnt = (Get-ChildItem -Recurse -File -Path $pat -Include *.jpg,*.jpeg,*.png -ErrorAction SilentlyContinue).Count
            if ($cnt -gt 0) {
                Copy-Sample -SourceGlob $pat -Dest $trainDest -Count $TrainPerClass
                Copy-Sample -SourceGlob $pat -Dest $valDest -Count $ValPerClass
                Copy-Sample -SourceGlob $pat -Dest $testDest -Count $TestPerClass
                $matched = $true
                break
            }
        }

        if (-not $matched) { Write-Warning "No images found for class '$cls' in downloaded datasets." }
    }

    Write-Host "DONE. Review data/images/train|val|test and adjust class names if needed."
}
finally {
    Pop-Location
}


