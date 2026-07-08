# DRVision: Deep Learning Framework for Early Detection and Severity Analysis of Diabetic Retinopathy

DRVision is an automated screening framework designed to classify the severity of Diabetic Retinopathy (DR) into 5 distinct clinical stages using retinal fundus images.

## 🚀 Framework Features
* **Core Architecture:** EfficientNet-B3 utilized via Transfer Learning.
* **Image Preprocessing:** Contrast Limited Adaptive Histogram Equalization (CLAHE), uniform resizing (512x512), and noise reduction.
* **Explainable AI:** Integrated Grad-CAM visualization to highlight specific lesion regions (microaneurysms, hemorrhages, exudates) influencing model predictions.

---

## 🏗️ End-to-End System Architecture

To maintain high pipeline transparency, the raw data flows through a strict optimization process before reaching the classification head:

<p align="center">
  <img src="system_architecture.png" alt="End-to-End System Architecture Diagram" width="100%">
</p>

---

## 📊 Performance Results
The model achieves an overall classification accuracy of ~84% on the test split.

<p align="center">
  <img src="performance_metrics.png" alt="Detailed Performance Metrics Dashboard" width="100%">
</p>

*Note: The high QWK score of 0.87 indicates strong ordinal agreement with expert clinical grading labels.*

---

## 🖼️ Visualizations & Model Interpretability

### Preprocessing & Model Pipeline
The framework processes raw retinal fundus images through an optimized pipeline involving black border cropping, uniform resizing to 512x512, and Contrast Limited Adaptive Histogram Equalization (CLAHE) to dramatically enhance the contrast of subtle microaneurysms and exudates before passing them to the fine-tuned EfficientNet-B3 network.

### Clinical Explainability via Grad-CAM
To establish clinical trust, DRVision integrates Gradient-weighted Class Activation Mapping (Grad-CAM). Instead of acting as a "black box," the framework outputs a localization heatmap highlighting the exact pathological regions (such as hemorrhages or lipid deposits) driving its severity prediction.

| Original Retinal Fundus Image | AI Prediction Heatmap (Stage 2 - 94.3% Confidence) |
| --- | --- |
| ![Original Image](images/original_fundus.png) | ![Grad-CAM Heatmap](images/gradcam_heatmap.png) |
