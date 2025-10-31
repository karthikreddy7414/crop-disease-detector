# PlantVillage Dataset Auto-Download and Setup Script
# Downloads from GitHub, extracts, organizes, and prepares data for training

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "PlantVillage Dataset Auto-Setup" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

# Configuration
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$DATA_DIR = Join-Path $PROJECT_ROOT "data\images\raw"
$PLANTVILLAGE_DIR = Join-Path $DATA_DIR "PlantVillage"
$ZIP_FILE = Join-Path $DATA_DIR "PlantVillage-Dataset-main.zip"
$GITHUB_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
$TEMP_CLONE_DIR = Join-Path $DATA_DIR "PlantVillage-Dataset-temp"

# Create directories
Write-Host "Creating directory structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $DATA_DIR | Out-Null
Write-Host "   Created: $DATA_DIR" -ForegroundColor Green

# Step 1: Clone from GitHub using git
Write-Host "`nDownloading PlantVillage Dataset from GitHub..." -ForegroundColor Yellow
Write-Host "   Repo: $GITHUB_REPO" -ForegroundColor Gray

# Check if git is available
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "   git not found. Install Git first!" -ForegroundColor Red
    Write-Host "   Download: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

try {
    if (Test-Path $TEMP_CLONE_DIR) {
        Write-Host "   Removing existing temp directory..." -ForegroundColor Yellow
        Remove-Item -Path $TEMP_CLONE_DIR -Recurse -Force
    }
    
    Write-Host "   Cloning repository (this may take a few minutes)..." -ForegroundColor Yellow
    Push-Location $DATA_DIR
    git clone $GITHUB_REPO PlantVillage-Dataset-temp
    Pop-Location
    
    Write-Host "   Download complete!" -ForegroundColor Green
}
catch {
    Write-Host "   Clone failed: $_" -ForegroundColor Red
    Write-Host "`nManual Download:" -ForegroundColor Yellow
    Write-Host "   1. Go to: https://github.com/spMohanty/PlantVillage-Dataset" -ForegroundColor Gray
    Write-Host "   2. Click Code -> Download ZIP" -ForegroundColor Gray
    Write-Host "   3. Save to: $DATA_DIR" -ForegroundColor Gray
    Write-Host "   4. Run this script again" -ForegroundColor Gray
    exit 1
}

# Step 2: Organize cloned data
Write-Host "`nOrganizing dataset structure..." -ForegroundColor Yellow

try {
    if (Test-Path $PLANTVILLAGE_DIR) {
        Write-Host "   Target directory exists, removing..." -ForegroundColor Yellow
        Remove-Item -Path $PLANTVILLAGE_DIR -Recurse -Force
    }
    
    # Create PlantVillage directory
    New-Item -ItemType Directory -Force -Path $PLANTVILLAGE_DIR | Out-Null
    
    # Copy color images from cloned repo
    $colorDir = Join-Path $TEMP_CLONE_DIR "raw\color"
    if (Test-Path $colorDir) {
        Write-Host "   Copying images..." -ForegroundColor Yellow
        Copy-Item -Path "$colorDir\*" -Destination $PLANTVILLAGE_DIR -Recurse -Force
        Write-Host "   Organization complete!" -ForegroundColor Green
    }
    else {
        # Alternative: copy root level directories if structure is different
        $subdirs = Get-ChildItem -Path $TEMP_CLONE_DIR -Directory | Where-Object { $_.Name -notlike "*.git*" -and $_.Name -ne "raw" }
        if ($subdirs) {
            Copy-Item -Path $subdirs.FullName -Destination $PLANTVILLAGE_DIR -Recurse -Force
            Write-Host "   Organization complete!" -ForegroundColor Green
        }
    }
    
    # Clean up temp directory
    if (Test-Path $TEMP_CLONE_DIR) {
        Remove-Item -Path $TEMP_CLONE_DIR -Recurse -Force
        Write-Host "   Cleaned up temp files" -ForegroundColor Green
    }
}
catch {
    Write-Host "   Organization failed: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Verify extraction
Write-Host "`nVerifying dataset structure..." -ForegroundColor Yellow

if (Test-Path $PLANTVILLAGE_DIR) {
    $subdirs = Get-ChildItem -Path $PLANTVILLAGE_DIR -Directory
    Write-Host "   Found $($subdirs.Count) disease classes" -ForegroundColor Green
    
    Write-Host "`n   Sample classes found:" -ForegroundColor Cyan
    $subdirs | Select-Object -First 10 | ForEach-Object {
        Write-Host "      - $($_.Name)" -ForegroundColor Gray
    }
    
    if ($subdirs.Count -gt 10) {
        Write-Host "      ... and $($subdirs.Count - 10) more" -ForegroundColor Gray
    }
}
else {
    Write-Host "   PlantVillage directory not found!" -ForegroundColor Red
    exit 1
}

# Step 4: Clean up ZIP file
Write-Host "`nCleaning up..." -ForegroundColor Yellow
if (Test-Path $ZIP_FILE) {
    Remove-Item -Path $ZIP_FILE -Force
    Write-Host "   Removed ZIP file" -ForegroundColor Green
}

# Step 5: Summary
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Dataset Setup Complete!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Summary:" -ForegroundColor Yellow
Write-Host "   Dataset location: $PLANTVILLAGE_DIR" -ForegroundColor Gray
Write-Host "   Total classes: $($subdirs.Count)" -ForegroundColor Gray
Write-Host "   Status: Ready for training" -ForegroundColor Gray

Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "   1. Prepare data:  python scripts/prepare_real_data.py" -ForegroundColor Cyan
Write-Host "   2. Train models:  python train_models.py" -ForegroundColor Cyan
Write-Host "   3. Start server:  python -m uvicorn src.main:app --reload" -ForegroundColor Cyan

Write-Host "`nReady to train with real plant disease images!" -ForegroundColor Green
Write-Host ""

