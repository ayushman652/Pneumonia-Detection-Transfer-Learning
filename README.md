# 🩺 Pneumonia Detection using Transfer Learning (CNN)

## 📌 Overview
This project implements a deep learning pipeline to detect pneumonia from chest X-ray images using Convolutional Neural Networks and transfer learning.

It focuses on comparing multiple architectures and optimizing performance through fine-tuning and data augmentation.

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
- Feature Extraction vs Fine-Tuning comparison  
- Data Augmentation to improve generalization  
- Batch Normalization and Dropout for regularization  

---

## 🏆 Final Model Performance
- **Validation Accuracy:** 96.5%  
- **Recall:** ~99%  
- High sensitivity for pneumonia detection  

---

## 📊 Model Comparison (Initial Results)
| Model        | Approach                     | Accuracy |
|-------------|-----------------------------|----------|
| VGG16       | Feature Extraction           | ~85%     |
| DenseNet121 | Fine-Tuning                 | ~85%     |
| ResNet50    | Fine-Tuning                 | ~63%     |

---

## 🔍 Key Insights
- Fine-tuning significantly improved performance over static feature extraction  
- Learning rate tuning reduced training instability  
- Data augmentation improved generalization  
- DenseNet121 showed strong robustness  

---

## 📁 Dataset
- Chest X-ray dataset (Kaggle Pneumonia Dataset)

---

## ▶️ How to Run
1. Install dependencies  
2. Load dataset  
3. Run notebook  

---

## 🎯 Objective
To build a reliable deep learning model for early detection of pneumonia from medical images.

---

## 📌 Future Improvements
- Deploy as a web application  
- Use EfficientNet / ensemble models  
- Add explainability (Grad-CAM)  

---

## 📬 Contact
- LinkedIn: https://www.linkedin.com/in/ayushman-singh-444902283/
