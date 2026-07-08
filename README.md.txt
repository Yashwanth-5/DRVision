# DRVision: Deep Learning Framework for Early Detection and Severity Analysis of Diabetic Retinopathy

DRVision is an automated screening framework designed to classify the severity of Diabetic Retinopathy (DR) into 5 distinct clinical stages using retinal fundus images.

## 🚀 Framework Features
* **Core Architecture:** EfficientNet-B3 utilized via Transfer Learning.
* **Image Preprocessing:** Contrast Limited Adaptive Histogram Equalization (CLAHE), uniform resizing (512x512), and noise reduction.
* **Explainable AI:** Integrated Grad-CAM visualization to highlight specific lesion regions (microaneurysms, hemorrhages, exudates) influencing model predictions.

## 📊 Performance Results
The model achieves an overall classification accuracy of ~84% on the test split.

| Performance Metric | Evaluation Value |
| --- | --- |
| Accuracy | 0.84 |
| Precision | 0.84 |
| Recall | 0.86 |
| F1-Score | 0.85 |
| Quadratic Weighted Kappa (QWK) | 0.87 |

*Note: The high QWK score of 0.87 indicates strong ordinal agreement with expert clinical grading labels.*