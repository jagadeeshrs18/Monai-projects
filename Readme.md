🧠 Brain Tumor Segmentation with SegResNet

Automatic brain tumor detection and segmentation using deep learning. Achieves 70-75% Dice score with a beautiful interactive web interface.

📋 Overview
This project uses SegResNet (Segmentation Residual Network) to automatically detect and segment brain tumors from MRI scans. The system identifies three tumor types:

🔴 Necrotic Core - Dead tissue
🟢 Edema - Swelling
🔵 Enhancing Tumor - Active tumor

Key Features:

🎯 70-75% Dice accuracy
⚡ < 5 second inference
🎨 25+ visualizations
💻 Interactive web app
📊 Comprehensive reports


 Quick Start
Installation
bash# Clone repository
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# Install dependencies
pip install -r requirements.txt
Run Web App
bashstreamlit run app.py
Visit http://localhost:8501 and upload a brain scan!

📦 Requirements
txttorch>=2.0.0
monai[all]>=1.3.0
streamlit>=1.28.0
nibabel>=5.0.0
matplotlib>=3.7.0
plotly>=5.17.0
opencv-python>=4.8.0
scipy>=1.11.0
pillow>=10.0.0
numpy>=1.24.0

💾 Dataset
BRATS 2020 - Brain Tumor Segmentation Challenge

369 training cases
4 MRI modalities per case
Multi-class annotations

Download: Kaggle BRATS 2020

🎓 Training
Train the Model
bashpython train_segresnet.py
Configuration:

Epochs: 45
Batch Size: 2 (adjust for GPU memory)
Learning Rate: 1e-4
Patch Size: 96×96×96
Training Time: ~2.5 hours on P100 GPU

Training Results
Best Validation Dice: 0.7234
- Necrotic Core: 0.7156
- Edema: 0.7589
- Enhancing Tumor: 0.6957

📊 Model Performance
ModelDice ScoreSpeedParametersBaseline UNet0.523s16MSegResNet0.70-0.754s18MImprovement+38%-25%+12%

Results
Performance Metrics
ModelDice ScoreParametersSpeedUNet (Baseline)0.5216M3sSegResNet0.70-0.7518M4sImprovement+38%+12%-25%

🖥️ Streamlit App Features
 Interactive Interface

Single image upload (PNG/JPG/NIfTI)
Real-time AI analysis
Black cyberpunk theme with animations
4 view tabs (Overview, 3D, Metrics, Report)

📊 Visualizations (25+)

Detection overlays
Confidence heatmaps
3D interactive plots
Contour maps
Statistical charts
Risk assessment
Downloadable reports

🎨 Screenshots
Show Image
Show Image

📁 Project Structure
brain-tumor-segmentation/
├── app.py                    # Streamlit web app
├── kagglenotebook.ipynb      # Training script
├── requirements.txt          # Dependencies
├── segresnet_complete.pth    # Trained model      
└── README.md                 # This file

🎯 Usage Examples
1. Train Your Own Model
pythonpython train_segresnet.py --epochs 45 --batch_size 2
2. Run Inference
pythonfrom model import SegResNet
import torch

model = SegResNet.load('models/segresnet_complete.pth')
prediction = model(input_image)
3. Launch Web App
bashstreamlit run app.py
# Upload image → Get analysis → Download report

🔧 Configuration
Edit training parameters in train_segresnet.py:
pythonCONFIG = {
    'num_epochs': 45,
    'batch_size': 2,
    'learning_rate': 1e-4,
    'patch_size': (96, 96, 96),
    'num_train_samples': 120,
}

📈 Results & Visualizations
Training Curves
Show Image
Prediction Examples
OriginalDetectionHeatmapShow ImageShow ImageShow Image

Local Deployment
bashstreamlit run app.py --server.port 8080

OUTPUT 

![WhatsApp Image 2025-12-07 at 14 12 06_e1ba8ef8](https://github.com/user-attachments/assets/bb68b36f-8922-477c-a39c-28f209ff8839)

![WhatsApp Image 2025-12-07 at 14 11 52_f753cf5a](https://github.com/user-attachments/assets/4fc509b5-c10c-4e04-ac90-70b210d8b06b)


🛠️ Troubleshooting
CUDA Out of Memory?
bash# Reduce batch size
python train_segresnet.py --batch_size 1
App Not Loading Model?
bash# Check model file exists
ls models/segresnet_complete.pth
Slow Training?
bash# Use smaller patch size
python train_segresnet.py --patch_size 64


📄 License
MIT License - see LICENSE file

main technologies :
BRATS Challenge - Dataset
MONAI - Medical imaging framework
Streamlit - Web app framework
PyTorch - Deep learning
Streamlit - Web app framework
PyTorch - Deep learning
