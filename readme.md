# CheXNet: Chest X-ray Disease Classification

A deep learning project for **automated multi-label chest X-ray disease classification** using transfer learning on the NIH ChestX-ray14 dataset.

## 📊 Project Overview

This repository implements a CheXNet-inspired model for detecting **14 thoracic diseases** from chest X-ray images. The project demonstrates transfer learning techniques, multi-label classification, and proper handling of class imbalance - key skills for computer vision roles.

## 🎯 Problem & Impact

**Challenge:** Automated detection of chest abnormalities to assist radiologists in early diagnosis.

**Dataset:** NIH ChestX-ray14 - 112,122 X-ray images labeled across 14 disease categories.

**Clinical Significance:** Early pneumonia detection from X-rays can significantly improve patient outcomes, yet manual review is time-intensive and variable.

**My Contribution:** Reproduced and extended the CheXNet architecture with enhanced training pipeline, uncertainty estimation, and test-time augmentation.

## 🏗️ Model Architecture

| Component | Details |
|-----------|---------|
| **Backbone** | DenseNet121 pretrained on ImageNet |
| **Task** | Multi-label classification (14 diseases) |
| **Output** | Sigmoid-activated logits per disease class |
| **Regularization** | Dropout, Weight Decay (1e-5) |

## 📈 Key Results

| Metric | Score | Baseline (Paper) |
|--------|-------|------------------|
| **Mean AUROC** | **0.6731** | 0.8417 |
| **Pneumonia F1-score** | **0.0550** | 0.435 |
| **AUROC 95% CI** | **[0.6573, 0.6885]** | - |

*Training for 20 epochs with early stopping; results may improve with longer schedules.*

## 🔬 Methodology

### Transfer Learning
- DenseNet121 initialized with ImageNet pretrained weights
- Final classifier replaced with 14-neuron layer for disease classes

### Class Imbalance Handling
- **Pos-weighted BCEWithLogitsLoss** - assigns higher weight to rare diseases
- Class counts from training set used to compute `pos_weight` per class

### Training Pipeline
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler:** ReduceLROnPlateau (factor=0.1, patience=1)
- **Early Stopping:** patience=3 epochs
- **Data Augmentation:** RandomResizedCrop, RandomHorizontalFlip
- **Test-Time Augmentation:** TenCrop averaging for improved stability

### Evaluation Metrics
- **Primary:** AUROC (micro-average and per-class)
- **Secondary:** F1-score (Pneumonia at optimal threshold)
- **Reliability:** Bootstrap 95% Confidence Intervals

## 📁 Project Structure

```text
chexnet/
├── README.md              # Project overview & results
├── requirements.txt       # Python dependencies
├── chexnet/               # Python package
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── densenet.py   # DenseNet121 architecture
│   │   └── resnet.py     # ResNet18 architecture
│   ├── dataset.py        # Dataset & splitting utilities
│   ├── transforms.py     # Image transforms (train/val/test)
│   ├── engine.py         # Training & evaluation loops
│   ├── evaluate.py       # Metrics (AUROC, F1, Bootstrap)
│   └── visualize.py      # Grad-CAM interpretability
├── notebooks/             # Original Jupyter notebooks
├── data/                  # Processed data directories
└── runs/                  # Experiment logs & checkpoints
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Build data splits (if not already present)
python - from chexnet.dataset import build_splits
build_splits("/path/to/chest-xray14", "/path/to/Data_Entry_2017.csv")

# 2. Train model
python - from chexnet.engine import train
train(
    path_dir_data="/path/to/images",
    path_train="train.csv",
    path_val="val.csv",
    arch="DENSE-NET-121",
    max_epochs=20,
    lr=1e-3
)

# 3. Evaluate on test set
python - from chexnet.engine import test
test(
    path_dir_data="/path/to/images",
    path_test="test.csv",
    arch="DENSE-NET-121",
    use_tencrop=True
)

# 4. Grad-CAM visualization
python - from chexnet.visualize import visualize_prediction
visualize_prediction(model, "/path/to/xray.jpg")
```

## 📦 Dependencies

See `requirements.txt` for full list. Key packages:
- **PyTorch** - Deep learning framework
- **torchvision** - Prebuilt models and transforms
- **scikit-learn** - AUROC and F1-score computation
- **matplotlib** - Plotting and visualization
- **Pillow** - Image handling
- **iterstrat** - Multilabel stratified splitting

## 👤 About Me

This project was developed as part of my computer vision portfolio demonstrating:
- Transfer learning and fine-tuning skills (DenseNet121 + ImageNet weights)
- Multi-label classification pipeline design
- Handling severe class imbalance with pos-weighted loss
- Model evaluation with statistical confidence intervals (bootstrap CI)
- Test-time augmentation (TenCrop) for improved reliability
- Reproducible research practices with modular code structure
- Model interpretability via Grad-CAM visualizations

Results can be further improved with extended training (50+ epochs) and hyperparameter optimization - typical for real-world CV projects where compute/resources are constrained.

## 🔧 Future Improvements

- Longer training schedule (50+ epochs with cosine annealing)
- Hyperparameter tuning (learning rate, batch size, augmentation)
- Advanced augmentation: CutMix, MixUp
- Alternative architectures: EfficientNet, ConvNeXt
- Grad-CAM disease region localization (included in `chexnet.visualize`)
- Mixed precision training for faster iteration
- Docker containerization for reproducible experiments