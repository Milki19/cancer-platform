# Cancer Platform

Reproducible machine learning pipeline for evaluating classical machine learning and deep learning models on heterogeneous oncology datasets.

This project was developed as part of my bachelor thesis:

**Evaluation and Comparison of Machine Learning Models on Heterogeneous Oncology Datasets**

The focus of the project is not only on achieving high classification metrics, but also on building a transparent and reproducible workflow that includes data preparation, quality control, data leakage detection, model training and evaluation.

---

## Project Motivation

Machine learning methods are increasingly used in medical data analysis, especially for image-based and tabular diagnostic support systems.

However, high accuracy alone is not enough in medical machine learning. The reliability of the results depends heavily on:

- how the data is prepared,
- whether there is data leakage between training and test sets,
- how stable the model results are,
- whether the full experiment can be reproduced.

This project explores these issues using multiple public oncology datasets.

---

## Datasets

The project uses four publicly available Kaggle datasets:

| Dataset | Type | Task | Samples Used |
|---|---|---|---|
| SIPaKMeD | Histopathology images | Cervical cell classification | 4,049 cropped images |
| LC25000 | Histopathology images | Lung / colon tissue classification | 25,000 images |
| RM1000 Lung | Histopathology images | Lung tissue classification | 15,000 images |
| Thyroid Recurrence | Tabular clinical data | Recurrence prediction | 364 cleaned rows |

Dataset sources:

1. SIPaKMeD Cervical Cancer Dataset  
   https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed

2. Lung and Colon Cancer Histopathological Images  
   https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

3. Lung Cancer Histopathological Images  
   https://www.kaggle.com/datasets/rm1000/lung-cancer-histopathological-images

4. Differentiated Thyroid Cancer Recurrence  
   https://www.kaggle.com/datasets/joebeachcapital/differentiated-thyroid-cancer-recurrence

---

## Pipeline Overview

The full workflow follows these steps:

```text
Raw data
    ↓
Data preparation and standardization
    ↓
Stratified train / validation / test split
    ↓
Baseline model training
    ↓
Stability experiments
    ↓
Quality control and leakage detection
    ↓
Leakage-free split generation
    ↓
Final model evaluation
    ↓
Result registry and reporting
