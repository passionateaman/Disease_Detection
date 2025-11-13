# Pneumonia Detection using Convolutional Neural Networks (CNN)

## Project Overview
This project presents an **AI-based system** for detecting **Pneumonia from chest X-ray images** using **Deep Learning**.  
By leveraging **Convolutional Neural Networks (CNNs)**, the model automatically learns visual features from radiographic images and classifies them as **Normal** or **Pneumonia**, reducing the need for manual interpretation by radiologists.

---

## Motivation
Pneumonia is one of the leading causes of death worldwide, particularly among children and the elderly.  
Manual diagnosis through X-rays can be **time-consuming, subjective, and prone to human error**, especially in resource-limited areas.  
This project aims to create a **reliable, automated, and scalable** solution that assists healthcare professionals in early and accurate diagnosis.

---

## ⚙️ Technologies Used
| Component | Description |
|------------|-------------|
| **Language** | Python |
| **Framework** | TensorFlow / Keras |
| **Libraries** | NumPy, OpenCV, Matplotlib |
| **Dataset** | Kaggle – Chest X-Ray (Pneumonia) |
| **Model Type** | Convolutional Neural Network (CNN) |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-Score |

---

## Dataset Description
**Dataset:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
- Total Images: ~5,863  
- Classes:
  - **Normal** – Healthy lungs  
  - **Pneumonia** – Bacterial/Viral infection  
- Train/Validation/Test Split:
  - 70% Train, 15% Validation, 15% Test

---

## Methodology

### 1. Data Preprocessing
- Resized all images to **150 × 150 pixels**.  
- Normalized pixel values (0–1).  
- Applied **augmentation** (rotation, flipping, brightness adjustment) to improve generalization.

### 2. CNN Architecture
- **3 Convolutional Layers** with ReLU activation  
- **Batch Normalization** and **MaxPooling** after each convolution  
- **Dropout Layers** for regularization  
- **Dense Layers** for final classification  
- **Sigmoid Activation** in output layer for binary classification

### 3. Model Compilation
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Metrics:** Accuracy

### 4. Model Training
- Trained for **15 epochs**  
- Used **ImageDataGenerator** for real-time augmentation  
- Visualized accuracy/loss curves after training

---

## 🧮 Model Performance

| Metric | Value |
|---------|-------|
| Training Accuracy | 98% |
| Validation Accuracy | 88% |
| Precision | 89% |
| Recall | 95% |
| F1-Score | 91% |

**Interpretation:**  
The CNN model effectively distinguishes between Pneumonia and Normal cases with high recall (few false negatives), making it suitable for medical screening.

---

##  Results Visualization
During training:
- Accuracy and loss curves remained stable → no overfitting.
- Model generalized well to unseen test data.
- Augmentation improved robustness to lighting and orientation variations.

---

## Future Scope
1. **Multi-disease detection:** Extend model to detect COVID-19, Lung Cancer, and Tuberculosis.  
2. **Mobile/Edge deployment:** Convert model for use in mobile healthcare apps.  
3. **Explainable AI (XAI):** Use Grad-CAM to visualize infected lung regions.  
4. **Integration:** Embed system into hospital cloud platforms for real-time diagnostics.  

---

## 📂 Repository Structure
