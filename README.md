# 🌱 Crop Disease Detection System

AI-powered image analysis system for identifying crop diseases with **76.2% accuracy** across 25 disease types.

## ✨ Features

- **Image-Based Disease Detection**: Upload crop photos for instant disease identification
- **High Accuracy**: 76.2% accuracy trained on 6,920+ real crop disease images
- **25 Disease Classes**: Identifies diseases across multiple crops
- **Web Interface**: User-friendly interface for easy disease analysis
- **Fast Processing**: Results in seconds

## 🎯 Supported Crops & Diseases

### Crops Supported:
- **Apple** - Scab, Black Rot, Cedar Apple Rust, Healthy
- **Corn (Maize)** - Common Rust, Cercospora Leaf Spot, Northern Leaf Blight, Healthy
- **Grape** - Black Rot, Esca, Leaf Blight, Healthy
- **Potato** - Early Blight, Late Blight, Healthy
- **Peach** - Bacterial Spot, Healthy
- **Pepper (Bell)** - Bacterial Spot, Healthy
- **Blueberry, Cherry, Orange, Raspberry** - Various diseases

**Total: 25 disease classifications**

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- pip package manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd DNA-crop

# Install dependencies
pip install -r requirements.txt

# Start the web server
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

### Usage

1. Open your browser to `http://127.0.0.1:8000`
2. Upload a crop image
3. Select the crop type
4. Click "Analyze Image"
5. Get instant disease prediction!

## 📊 Model Performance

| Model Type | Accuracy | Training Data |
|------------|----------|---------------|
| **Image Classifier** | **76.2%** | 6,920 real images |
| Gradient Boosting | 71.7% | 6,920 real images |
| SVM | 69.9% | 6,920 real images |
| Fusion Model | 74.2% | Combined features |

**Training Dataset**: PlantVillage - 6,920 images across 25 disease classes

## 🏗️ Project Structure

```
DNA-crop/
├── src/
│   ├── api/                    # FastAPI routes
│   │   └── routes/
│   │       ├── image_processing.py   # Image analysis API
│   │       ├── dna_analysis.py       # DNA analysis API
│   │       └── disease_prediction.py # Prediction API
│   ├── image_processing/       # Image processing modules
│   ├── dna_analysis/          # DNA analysis modules
│   ├── models/                # Model implementations
│   ├── training/              # Training pipeline
│   └── main.py                # Main application
├── models/
│   └── real_trained/          # Trained models (76.2% accuracy)
│       ├── image_classifier_best.joblib
│       ├── fusion_model.joblib
│       └── metadata.json
├── static/
│   └── index.html             # Web interface
├── data/
│   ├── dna/                   # DNA data
│   └── images/                # Image datasets
├── scripts/                   # Helper scripts
├── requirements.txt           # Python dependencies
├── config.py                  # Configuration
└── README.md                  # This file
```

## 🔧 API Endpoints

### Image Analysis
```bash
POST /api/image/preprocess
```
**Parameters:**
- `image`: Image file (JPG, PNG)
- `crop_type`: Crop type (apple, corn, grape, etc.)

**Returns:**
- Disease name
- Confidence score
- Severity level
- Treatment recommendation

### Health Check
```bash
GET /health
GET /api/image/health
GET /api/dna/health
GET /api/prediction/health
```

## 💻 Tech Stack

- **Backend**: FastAPI, Python 3.12
- **ML**: scikit-learn, Random Forest, SVM, Gradient Boosting
- **Image Processing**: OpenCV, Pillow
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: SQLite, SQLAlchemy

## 📦 Model Training

The system includes a training script for retraining models with new data:

```bash
python train_with_real_images.py
```

**Training Parameters:**
- Image size: 128x128 pixels
- Features: Color histograms + texture features (29 dimensions)
- Max images per class: 300
- Train/Test split: 80/20

## 🎓 Training Data

Models are trained on the **PlantVillage Dataset**:
- Source: https://github.com/spMohanty/PlantVillage-Dataset
- 54,000+ plant disease images
- Professional quality, controlled conditions
- Multiple crops and disease types

## 🛠️ Development

### Run in Development Mode
```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

### Run in Production Mode
```bash
python start_production.py
```

## 📝 Configuration

Edit `config.py` to customize:
- Server host and port
- Model paths
- Logging settings
- Performance parameters

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Add more crop types
- Improve accuracy with more training data
- Add real DNA sequence analysis
- Mobile app development
- Batch processing capabilities

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **PlantVillage Dataset** for providing high-quality plant disease images
- **scikit-learn** for machine learning tools
- **FastAPI** for the web framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Status**: Production Ready ✅  
**Last Updated**: November 2024  
**Model Version**: 1.0 (Real Image Training)
