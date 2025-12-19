# Pneumonia Detection using Deep Transfer Learning

This project evaluates three Deep Learning architectures—**VGG16, ResNet50, and DenseNet121**—for the binary classification of Pneumonia from Chest X-Ray images. The study focuses on comparing the efficiency of **Static Feature Extraction** versus **Two-Phase Fine-Tuning**.

## 🚀 Key Results
| Model | Training Strategy | Accuracy | Recall | Key Insight |
| :--- | :--- | :--- | :--- | :--- |
| **VGG16** | Frozen Base (Feature Extraction) | **85%** | **96%** | High efficiency, low training cost. |
| **DenseNet121** | Fine-Tuned (Unfrozen Top 30) | **85%** | **96%** | Robust to image variations due to heavy augmentation. |
| **ResNet50** | Fine-Tuned (Unfrozen Top 100) | ~63% | - | Suffered from optimization instability. |

## 🛠️ Model Configurations

All models were implemented using **TensorFlow/Keras**.

### 1. Global Settings
* **Input Shape:** `(224, 224, 3)`
* **Batch Size:** 32
* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Metrics:** Accuracy, Recall, AUC

### 2. Architecture-Specific Configs

#### **VGG16 (The "Frozen" Model)**
* **Base:** VGG16 (ImageNet weights), non-trainable.
* **Head:** `Flatten` $\rightarrow$ `Dense(256, ReLU)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Output(Sigmoid)`.
* **Learning Rate:** $10^{-4}$

#### **DenseNet121 (The "Robust" Model)**
* **Base:** DenseNet121 (ImageNet weights).
* **Training Strategy:** 2-Phase Training.
    * *Phase 1:* Train Head only.
    * *Phase 2:* Unfreeze top 30 layers.
* **Head:** `GlobalAveragePooling` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dropout(0.5)`.
* **Augmentation:** Rotation (15°), Zoom (15%), Width/Height Shift, Shear, Brightness.

## 📦 Requirements
To run this project, you will need the following libraries:
```python
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
