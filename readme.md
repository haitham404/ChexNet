# ChestX-ray14 Disease Classification

A deep learning project for automated **multi-label chest X-ray disease classification** using **transfer learning** on the **NIH ChestX-ray14 dataset**.



## Overview

This project aims to detect **14 thoracic diseases** from chest X-ray images using **DenseNet121** pretrained on **ImageNet**.



## Key Features

| Component                | Description                        |
| ------------------------ | ---------------------------------- |
| Dataset                  | NIH ChestX-ray14                   |
| Task                     | Multi-label Classification         |
| Architecture             | DenseNet121                        |
| Transfer Learning        | ImageNet Pretrained Weights        |
| Loss Function            | BCEWithLogitsLoss                  |
| Class Imbalance Handling | Class-Weighted Loss (`pos_weight`) |
| Evaluation Metric        | AUROC (Primary)                    |
| Additional Metric        | Pneumonia F1-score                 |
| Test-Time Augmentation   | TenCrop                            |
| Uncertainty Estimation   | Bootstrap 95% Confidence Interval  |



## Dataset Split

| Split      | Images |
| ---------- | ------ |
| Train      | 77,921 |
| Validation | 8,603  |
| Test       | 25,596 |

Patient-level splitting was applied to prevent **data leakage**.



## Results

| Metric             | Score                |
| ------------------ | -------------------- |
| Mean AUROC         | **0.6731**           |
| Pneumonia F1-score | **0.0550**           |
| Bootstrap 95% CI   | **[0.6573, 0.6885]** |



## Training Strategy

* Transfer Learning with **DenseNet121**
* Class-weighted **Binary Cross Entropy**
* **Adam Optimizer**
* Learning Rate Scheduling
* Early Stopping
* Model Checkpointing
* **TenCrop Test-Time Augmentation**



## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib



## Future Improvements

* Longer training schedule
* Hyperparameter tuning
* Advanced augmentation techniques
* Alternative architectures (**EfficientNet**, **ConvNeXt**)
* Improved handling of severe class imbalance
* loclization of disease regions using **Grad-CAM** or **attention mechanisms**


