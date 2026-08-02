# 🧠 Large-Scale Student Mental Health Detection System

An advanced NLP pipeline for detecting mental health conditions in student populations using deep learning with transformer-based architecture and 15,000+ synthetic samples.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [Vocabulary Building](#-vocabulary-building)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements a comprehensive mental health detection system using advanced NLP techniques. It leverages transformer-based architectures to classify student text responses into five mental health categories:

1. **Depression (Severe)** - Severe depressive symptoms
2. **Depression (Moderate)** - Moderate depressive symptoms
3. **Anxiety (Severe)** - Severe anxiety symptoms
4. **Anxiety (Moderate)** - Moderate anxiety symptoms
5. **Normal** - No mental health concerns

---

## ✨ Features

- **Large-Scale Dataset Generation**: 15,000+ synthetic samples with realistic patterns
- **Advanced Text Preprocessing**: Lemmatization, custom stopwords, noise removal
- **Large Vocabulary**: 10,000+ word vocabulary with subword tokenization
- **Transformer Architecture**: Multi-head attention with positional encoding
- **Data Augmentation**: Mixup, token dropout, and replacement techniques
- **Ensemble Ready**: Modular design for model ensembling
- **Weighted Sampling**: Handles class imbalance effectively
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Comprehensive Evaluation**: Classification reports, confusion matrices, and visualizations
- **Production Ready**: Save/load models, preprocessors, and encoders

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Framework** | PyTorch |
| **NLP** | NLTK, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Logging** | Python logging |
| **Serialization** | Pickle, Joblib |

---

## 📦 Installation

### Option 1: Google Colab

```python
# Install dependencies
!pip install torch nltk scikit-learn pandas numpy matplotlib seaborn tqdm

# Clone the repository
!git clone <repository-url>
cd <project-directory>

# Run the script
python mental_health_detection.py
