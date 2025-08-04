
# Training Script for VGG16 Image Classifier with Early Stopping

This project implements an image classifier using a pre-trained VGG16 model, fine-tuned for image classification. It includes early stopping to prevent overfitting and saves the best performing model.

**Author**: Dr. Amit Chougule  
**Date**: 25/07/2025

---

## 🔧 Features

- Transfer learning using pretrained VGG16
- Custom classifier for image classification
- Data augmentation and normalization
- Early stopping to prevent overfitting
- Training and validation accuracy tracking
- Accuracy plot saved after training

---

## 🗂️ Directory Structure

```

project/
│
├── Data_Dir/
│ ├── train/
│ │ ├── class_0/
│ │ └── class_1/
│ └── val/
│ ├── class_0/
│ └── class_1/
│
├── training/
│ └── train.py # Training script
│
├── testing/
│ └── test.py # Testing script
│
├── vgg16_best_model.pth # Saved best model (after training)
├── vgg16_image_classifier.pth # Final model (last epoch)
└── accuracy_plot.png # Accuracy plot

```

---

## 📦 Requirements

Install dependencies using `pip`:

```bash
pip install torch torchvision matplotlib
````

Make sure you have access to a CUDA-compatible GPU for best performance, although the code will run on CPU as well.

---

## 📁 Dataset Format

The code expects the dataset to be organized as follows:

```
<Data_Dir>/
├── train/
│   ├── class_0/
│   └── class_1/
└── val/
    ├── class_0/
    └── class_1/

```

Replace `<Data_Dir>` in the script with the actual path to your dataset.

---

## 🚀 How to Run

1. Update the `data_dir` variable in the script to point to your dataset.
2. Run the script:

```bash
python train.py
```

---

## 🧠 Model Overview

- **Base Model**: Pretrained VGG16 (`torchvision.models.vgg16`)
- **Feature Extractor**: Frozen
- **Classifier Modified**: `nn.Linear(4096, 2)` for binary classification
- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `Adam (LR = 1e-4)`
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Early Stopping Patience**: 5 epochs without improvement in validation accuracy

---

## 📊 Output

* **Best Model**: Saved as `vgg16_best_model.pth`
* **Final Model (Last Epoch)**: Saved as `vgg16_image_classifier.pth`
* **Accuracy Plot**: Saved as `accuracy_plot.png`

---


