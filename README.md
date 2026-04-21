# 🩺 Pneumonia Detection using Transfer Learning (CNN)

## 📌 Overview
This project focuses on detecting pneumonia from chest X-ray images using deep learning. It implements a complete pipeline for medical image classification using Convolutional Neural Networks (CNNs) and transfer learning.

---

## 🚀 Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy, Matplotlib  

---

## 🧠 Models Implemented
- VGG16  
- ResNet50  
- DenseNet121  

---

## ⚙️ Approach
- Transfer Learning using pre-trained CNN models  
- Comparison of:
  - Feature Extraction (frozen layers)  
  - Fine-Tuning (unfreezing top layers)  
- Data Augmentation for improved generalization  
- Batch Normalization and Dropout for regularization  

---

## 📊 Results

### 🔹 Initial Results (Before Optimization)
- Accuracy: ~85%  
- Observed instability in deeper architectures  

### 🔹 Final Model Performance
- **Validation Accuracy:** 96.5%  
- **Recall:** ~99% (high sensitivity for pneumonia detection)  

---

## 🏆 Key Insights
- Fine-tuning significantly improved performance over static feature extraction  
- Data augmentation helped reduce overfitting  
- Learning rate tuning stabilized training and improved convergence  
- DenseNet121 provided better generalization compared to VGG16  

---

## 📁 Dataset
- Chest X-ray dataset (Kaggle Pneumonia Dataset)

---

## ▶️ How to Run
1. Install required libraries  
2. Load dataset  
3. Run training notebook/script  

---

## 🎯 Objective
To build an accurate and reliable deep learning model for early detection of pneumonia using medical imaging.

---

## 📌 Future Improvements
- Deploy as a web application  
- Use advanced architectures (EfficientNet, ensembles)  
- Improve explainability using Grad-CAM  

---

## 📬 Contact
- LinkedIn: https://www.linkedin.com/in/ayushman-singh-444902283/
