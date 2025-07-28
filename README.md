# FaceRead-Facial-Emotion-Detection-using-FER-2013


Facial Emotion Detection is a computer vision task that enables machines to recognize human emotions through facial expressions. This project builds a deep learning model using a custom Convolutional Neural Network (CNN) to classify facial emotions into seven categories.

---

## 📌 Project Overview

- **Objective**: Automatically detect emotions from grayscale facial images.
- **Classes**:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- **Approach**:
  - Custom-built CNN trained on an augmented dataset
  - Evaluated using standard classification metrics

---

## 🗂 Dataset Description

- **Dataset Name**: [FER-2013 - Facial Expression Recognition Challenge](https://www.kaggle.com/datasets/msambare/fer2013)
- **Source**: Kaggle
- **Format**: 48×48 pixel grayscale images (converted to image tensors for training)
- **Total Classes**: 7
- **Splits**:
  - **Original Split**: 90% Train / 5% Validation / 5% Test
  - **Updated Split**: 80% Train / 10% Validation / 10% Test
- **Data Augmentation**:
  - Horizontal Flipping
  - Random Rotation
  - Zoom
  - Applied **only on training data**

---

## 🧠 Model Architecture

The model is a custom CNN composed of:

### 🔍 Feature Extraction
- **Block 1**:
  - 2 Convolutional layers (64 filters)
  - Batch Normalization + ReLU + MaxPooling + Dropout
- **Block 2**:
  - 2 Convolutional layers (128 filters)
  - Batch Normalization + ReLU + MaxPooling + Dropout
- **Block 3**:
  - 2 Convolutional layers (256 filters)
  - Batch Normalization + ReLU + MaxPooling + Dropout

### 🧮 Classification
- Fully Connected Layer (512 neurons) + ReLU + Dropout
- Output Layer (7 neurons) with Softmax Activation

---

## ⚙️ Training Configuration

- **Input Shape**: (1, 48, 48) grayscale image
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Evaluation Metrics**:
  - Accuracy
  - Confusion Matrix
  - Precision / Recall / F1-Score
  - ROC Curve

---

## 📊 Results

The model achieved strong performance across all metrics:

| Metric         | Description                      |
|----------------|----------------------------------|
| Accuracy       | High overall classification rate |
| Confusion Matrix | Indicates class-wise performance |
| ROC Curve      | Multiclass ROC analysis          |
| F1-Score       | Balanced performance across classes |

---

## 📈 Visualizations

- ✅ Confusion Matrix
- ✅ ROC Curve (Multi-class)
- ✅ Classification Report
- ✅ Training and Validation Accuracy 
---

## 🧾 Conclusion

This project successfully demonstrates a custom deep learning approach for emotion classification using grayscale facial images. The use of data augmentation significantly enhanced model generalization. The CNN model performs well on both validation and test data, confirming its robustness.

---

**Project Guide**: Br. Tamal Maharaj  
**Institute**: Ramakrishna Mission Vivekananda Educational and Research Institute (RKMVERI), Belur Math, Howrah

---
