# AI600_PA2
Implementation of a high-performance Multi-Layer Perceptron (MLP) for the Quick, Draw! image classification challenge.
# Quick, Draw! Image Classification (MLP Challenge)
**Course:** CS 600 - Deep Learning (Spring 2026)  
**Student:** Shujat Ali Khan (Roll: 25280086)

## Project Overview
This project involves building and optimizing a Multi-Layer Perceptron (MLP) to classify hand-drawn sketches from Google's "Quick, Draw!" dataset. The challenge was to achieve the highest possible accuracy on 15 distinct classes while strictly adhering to a **3 million parameter limit** and a **40-epoch training cap**.

### Final Performance
- **Validation Accuracy:** 81.33%
- **Total Parameters:** 2,519,055
- **Architecture:** 5-Layer Deep MLP

---

## Model Architecture: UltimateMLP V2
The final "Champion" model utilizes several modern deep learning techniques to maximize performance within the parameter budget:

1. **GELU Activations:** Used Gaussian Error Linear Units to prevent dead neurons and improve gradient flow during deep training.
2. **Batch Normalization:** Applied after every linear layer to stabilize internal covariate shift and accelerate convergence.
3. **Heavy Regularization:** Employed aggressive Dropout (up to 50%) and Weight Decay (L2 regularization) to prevent the model from memorizing the 784-dimensional pixel space.
4. **Learning Rate Scheduling:** Used `ReduceLROnPlateau` to dynamically adjust the step size, allowing the model to settle into local minima effectively.



---

## Repository Structure
```text
.
├── 25280086_PA2.ipynb     # Main Jupyter Notebook with training/inference code
├── submission.txt         # Final predictions for the leaderboard portal
├── Report/
│   ├── report.pdf         # Final LaTeX technical report
│   └── confusion_matrix.png # Generated visual analysis
└── .gitignore             # Configured to exclude heavy .npy/.npz data files
