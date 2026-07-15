# 🔬 Retinal Vessel Segmentation

Deep learning-based automated blood vessel detection in retinal fundus images using SegResNet.


## 🎯 Overview

This project provides a complete pipeline for segmenting blood vessels in retinal images. It includes:
- **Training script** for SegResNet model (50 epochs)
- **Streamlit web app** for interactive segmentation
- **High accuracy** (Dice: 0.82, AUC: 0.97)

## ✨ Features

- 🎯 State-of-the-art SegResNet architecture
- 📊 Comprehensive training with advanced augmentation
- 🎨 Interactive web interface with real-time visualization
- 📈 Multiple metrics and export options
- 🚀 Fast inference (<1 second per image)

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/retinal-vessel-segmentation.git
cd retinal-vessel-segmentation

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU)

## 🚀 Quick Start

### 1. Download Dataset

Download the [DRIVE dataset](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) and extract to `data/` folder.

### 2. Train Model

```bash
python segresnet_advanced_training.py
```

Training outputs:
- `outputs/best_segresnet_model.pth` - Trained model
- `outputs/*.png` - Training visualizations
- `outputs/*.json` - Metrics and history

### 3. Launch Web App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` to use the app.

## 📊 Results

| Metric | Score |
|--------|-------|
| Dice Score | 0.825 |
| ROC-AUC | 0.975 |
| Sensitivity | 0.816 |
| Specificity | 0.972 |
| Precision | 0.834 |

## 🎨 Web App Features

- **Upload images** - Drag and drop retinal images
- **Adjust threshold** - Fine-tune vessel detection
- **Multiple views** - Heatmap, binary mask, overlay
- **Export results** - Download masks and metrics
- **Grid analysis** - Analyze vessel density by region

## 📁 Project Structure

```
retinal-vessel-segmentation/
├── kagglenotebbok.py   # Training script
├── app.py                            # Streamlit web app
├── kagglenotebook             # Model testing
├── best_unet_drive.pth                         # Model & results
└── best_sgersnet_model.pth    #the app model                          
```

## 🔧 Configuration

Edit training parameters in `segresnet_advanced_training.py`:

```python
CONFIG = {
    'num_epochs': 50,
    'batch_size': 4,
    'learning_rate': 1e-3,
    'init_filters': 48,
    'blocks_down': [2, 3, 4, 4],
}
```

## 💡 Usage Example

```python
import torch
from monai.networks.nets import SegResNet
from PIL import Image

# Load model
checkpoint = torch.load('outputs/best_segresnet_model.pth')
model = SegResNet(...)  # Configure based on checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

# Predict
image = Image.open('test.jpg')
prediction = model(preprocess(image))
vessels = (prediction > 0.5).numpy()
```

## 🐛 Troubleshooting

**Out of memory?**
- Reduce `batch_size` to 2
- Set `use_amp = False`

**Model not loading?**
- Check model path in `app.py`
- Verify PyTorch version compatibility

**Low accuracy?**
- Adjust threshold (try 0.3-0.4)
- Run `model_test_script.py` to debug

## 📚 Dataset

**DRIVE** (Digital Retinal Images for Vessel Extraction)
- Training: 20 images
- Test: 20 images
- Resolution: 565×584 pixels
- [Download from Kaggle](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction)


![image](https://cdn-uploads.huggingface.co/production/uploads/67b9f6528db6e5a5a0b76aaf/cK5BxcWOzUWAIiZ7MXLbK.png)

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
