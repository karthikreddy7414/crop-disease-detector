"""
Automated PlantVillage Dataset Download Script
Downloads and organizes the PlantVillage dataset for crop disease training
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "images" / "raw"
PLANTVILLAGE_URL = "http://download1580.mediafire.com/abcdefghijkl/plantvillage.zip"  # Placeholder
GITHUB_REPO = "spMohanty/PlantVillage-Dataset"

def download_file(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

def download_plantvillage_manual():
    """
    Provides manual download instructions for PlantVillage
    (Most reliable method)
    """
    print("=" * 60)
    print("🌱 PLANTVILLAGE DATASET DOWNLOAD")
    print("=" * 60)
    print()
    print("📥 MANUAL DOWNLOAD INSTRUCTIONS:")
    print()
    print("Option 1: Kaggle (Recommended)")
    print("  1. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("  2. Click 'Download' button")
    print("  3. Extract ZIP to:", DATA_DIR / "PlantVillage")
    print()
    print("Option 2: GitHub")
    print("  1. Go to: https://github.com/spMohanty/PlantVillage-Dataset")
    print("  2. Click 'Code' → 'Download ZIP'")
    print("  3. Extract to:", DATA_DIR / "PlantVillage")
    print()
    print("Option 3: Direct Website")
    print("  1. Go to: http://plantvillage.psu.edu/")
    print("  2. Navigate to dataset download section")
    print("  3. Download and extract")
    print()
    print("=" * 60)

def setup_directory_structure():
    """Create directory structure for datasets"""
    directories = [
        DATA_DIR,
        DATA_DIR / "PlantVillage",
        DATA_DIR / "Rice",
        DATA_DIR / "Tomato",
        DATA_DIR / "Wheat",
        DATA_DIR / "Corn",
        DATA_DIR / "Potato",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory structure in: {DATA_DIR}")

def verify_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        print("✅ Kaggle API credentials found")
        return True
    else:
        print("⚠️  Kaggle API not configured")
        print("   Setup instructions:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token'")
        print("   3. Save kaggle.json to: ~/.kaggle/")
        return False

def download_via_kaggle():
    """Download using Kaggle API if available"""
    try:
        import kaggle
        
        dataset = "abdallahalidev/plantvillage-dataset"
        print(f"📥 Downloading dataset: {dataset}")
        
        kaggle.api.dataset_download_files(
            dataset,
            path=DATA_DIR / "PlantVillage",
            unzip=True
        )
        
        print("✅ Download complete!")
        return True
        
    except ImportError:
        print("❌ Kaggle package not installed")
        print("   Install: pip install kaggle")
        return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    """Main download function"""
    print("\n🌱 PLANT DISEASE DATASET DOWNLOADER\n")
    
    # Setup directories
    setup_directory_structure()
    
    # Check Kaggle
    kaggle_available = verify_kaggle_setup()
    
    # Try Kaggle download first
    if kaggle_available:
        try:
            import kaggle
            print("\n📥 Attempting Kaggle download...")
            if download_via_kaggle():
                print("\n✅ SUCCESS! Dataset downloaded via Kaggle")
                print(f"   Location: {DATA_DIR / 'PlantVillage'}")
                return
        except ImportError:
            pass
    
    # Fall back to manual instructions
    print("\n📋 Kaggle download not available, showing manual instructions...")
    download_plantvillage_manual()
    
    print("\n📝 NEXT STEPS:")
    print("   1. Download dataset manually using instructions above")
    print("   2. Extract to: data/images/raw/PlantVillage/")
    print("   3. Run: python scripts/prepare_real_data.py")
    print("   4. Run: python train_models.py")

if __name__ == "__main__":
    main()

