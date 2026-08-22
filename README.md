# Medical Image Analysis: Pneumonia Classification with Interpretability

## 📌 Project Overview
This repository implements an end-to-end Deep Learning pipeline in **PyTorch** for classifying chest X-ray images into **NORMAL** and **PNEUMONIA** cases. It leverages **Transfer Learning (ResNet18)** for feature extraction and integrates **Grad-CAM** to provide visual explainability for clinical decision support.

---

## 🏗️ Architecture & Pipeline
1. **Data Preprocessing & Augmentation:** Resizing to 224x224, Random Horizontal Flips, Tensor conversion, and ImageNet normalization.
2. **Transfer Learning:** Fine-tuning a pre-trained `ResNet18` model by replacing the classification head to output 2 logits.
3. **Training & Optimization:** Cross-Entropy Loss optimization using `Adam` optimizer ($lr=0.001$).
4. **Explainability (Grad-CAM):** Extracting gradient-weighted activation maps from `model.layer4[-1]` to visualize pathological attention zones.

---

## 🛠️ Tech Stack
* **Framework:** PyTorch, Torchvision
* **Interpretability:** `pytorch-gradcam`
* **Data Processing & Visualization:** OpenCV, Matplotlib, NumPy
* **Environment:** Google Colab (T4 GPU)

---

## 🚀 How to Run
1. Open `Pneumonia_Detection_ResNet18.ipynb` in Google Colab.
2. Provide your Kaggle API key to download the dataset.
3. Execute all cells sequentially.
