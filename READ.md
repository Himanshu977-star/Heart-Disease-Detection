# 🫀 CheXchoNet – Cardiac Diagnosis with Grad-CAM

CheXchoNet is a deep learning–based cardiac screening system built using **EfficientNetB2** and **Grad-CAM** for visual explainability.  
The model predicts the likelihood of **SLVH** and **DLV** conditions from medical images and highlights the regions that influenced the decision.

---

## 🔬 Model Training Overview

- Original dataset size: **~70,000 images**
- Due to computational constraints, the model was trained on **10,000 images**
- To handle **class imbalance**:
  - **6,000** images were sampled as baseline data
  - **4,000** additional images containing **disease-positive cases** were intentionally added
- This strategy helped improve learning for underrepresented cardiac conditions while keeping training efficient.

---

## 🧠 Model Architecture

- Backbone: **EfficientNetB2** (`include_top=False`)
- Classification head:
  - Global Average Pooling
  - Dense (256, ReLU)
  - Dropout (0.2)
  - Dense (2, Sigmoid)

The model outputs **independent probabilities** for:
- **SLVH**
- **DLV**

---

## 🔍 Explainability with Grad-CAM

Grad-CAM is used to visualize **where the model focuses** while making predictions, improving interpretability and trust in results.

Users can select which condition (SLVH or DLV) to visualize.

---

## ⚠️ Thresholding Logic

- Threshold used: **0.3**
- If probability ≥ 0.3 → **Abnormal**
- If both conditions < 0.3 → **Normal**

This setup is designed for **screening purposes**, not definitive diagnosis.

---

## 🚀 Live Demo

Upload an image, select the condition, and view:
- Prediction scores
- Normal / Abnormal status
- Grad-CAM heatmap

---

## ⚕️ Disclaimer

This project is intended for **research and educational purposes only**  
and **must not be used as a clinical diagnostic tool**.

Always consult qualified medical professionals for medical decisions.
