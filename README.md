# AI600_PA2
Implementation of a high-performance Multi-Layer Perceptron (MLP) for the Quick, Draw! image classification challenge.
# Quick, Draw! Image Classification Challenge
**Course:** CS 600 - Deep Learning (Spring 2026)  
**Student:** Shujat Ali Khan (Roll: 25280086)

## Project Overview
This repository contains the implementation of a Multi-Layer Perceptron (MLP) pipeline to classify hand-drawn sketches from the Google "Quick, Draw!" dataset across 15 distinct classes. The project follows an iterative architectural design process to maximize accuracy while staying within a **3-million parameter limit** and a **40-epoch training constraint**.

### Final Performance Summary
* **Validation Accuracy:** 81.33%
* **Model Parameters:** 2,519,055
* **Training Duration:** 40 Epochs

---

## Model Evolution
The project documents the development of three specific architectures:

1. **PancakeMLP (Baseline):** A shallow network used to establish initial performance metrics and verify the data pipeline.
2. **TowerMLP (Intermediate):** A deeper architecture designed to capture more complex spatial hierarchies, achieving approximately 74% accuracy.
3. **ChampionMLP (Final Submission):** An optimized, highly regularized 5-layer model that achieved the final 81.33% accuracy.

### ChampionMLP Key Features
* **GELU Activations:** Implemented for better gradient flow and to prevent dead neurons during deep training.
* **Batch Normalization:** Applied after linear layers to stabilize internal covariate shift and accelerate convergence.
* **Aggressive Regularization:** Utilized 50% Dropout and L2 Weight Decay ($1 \times 10^{-4}$) to mitigate severe overfitting observed in earlier iterations.
* **Dynamic Learning Rate:** Employed a `ReduceLROnPlateau` scheduler to fine-tune weights as the model approached convergence.



---

## Repository Structure
```text
.
├── 25280086_PA2.ipynb     # Final Jupyter Notebook with full model history
├── submission.txt         # Comma-separated predictions for the leaderboard
├── .gitignore             # Configured to exclude heavy .npz/.npy data files
└── README.md              # Project documentation
