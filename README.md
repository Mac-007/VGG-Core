# 🧠 VGG16-Based Image Classification Framework

**Author**: Dr. Amit Chougule  
**Date**: 25/07/2025

---

VGG16-based image classification system with a modular PyTorch design. Integrates early stopping to minimize overfitting and robust generalization on domain-specific datasets. Includes standalone testing scripts for batch inference and performance evaluation.

---

## 🔍 Key Contributions & Importance

- **Modular VGG16 Framework**: Offers a clean, modular implementation of VGG16 for image classification using PyTorch, easily extensible to different domains and tasks.

- **Research-Ready Pipeline**: Implements a robust training loop with early stopping, validation tracking, and model checkpointing — aligning with best practices in deep learning research.

- **Generalization on Domain-Specific Data**: Fine-tuning on custom datasets allows the pretrained VGG16 backbone to adapt effectively while preserving transferable features, ensuring strong performance with limited data.

- **Overfitting Mitigation via Early Stopping**: Integrates early stopping based on validation performance, making the pipeline robust against overfitting and noisy datasets.

- **Inference-Ready Deployment**: Includes a standalone testing script with confidence scoring, supporting practical batch inference across varied image formats.

---

## 🔧 Features

- Transfer learning using pretrained VGG16 (from `torchvision`)
- Modular training pipeline using PyTorch
- Custom classifier head for binary classification
- Early stopping to prevent overfitting
- Data augmentation and normalization
- Accuracy and loss tracking with visualization
- Standalone testing script for inference and confidence scoring
- Model saving (best and final)

---


## 🗂️ Project Structure

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
├── testing/
│ └── test.py # Testing script
├── vgg16_best_model.pth # Saved best model (after training)
├── vgg16_image_classifier.pth # Final model (last epoch)
└── accuracy_plot.png # Accuracy plot

```

---

## 📦 Installation

Install the required Python libraries:

```bash
pip install torch torchvision matplotlib pillow
````

For best performance, use a CUDA-compatible GPU, although CPU is supported.

---

## 📁 Dataset Format

Structure your dataset as follows:

```
<Data_Dir>/
├── train/
│   ├── class_0/
│   └── class_1/
└── val/
    ├── class_0/
    └── class_1/
```

Update the `data_dir` variable in `train.py` accordingly.

---

## 🚀 Training

To train the model:

1. Navigate to the `training/` directory.
2. Run the training script:

```bash
python train.py
```

During training:

* The model uses early stopping (patience = 5)
* Best model is saved as `vgg16_best_model.pth`
* Final model is saved as `vgg16_image_classifier.pth`
* Accuracy plot saved as `accuracy_plot.png`

---

## 🔍 Testing

To test images in a folder:

```bash
python test.py --model_path ../vgg16_best_model.pth --image_folder ../test_images/
```

**Arguments:**

* `--model_path`: Path to `.pth` model file
* `--image_folder`: Path to folder containing test images (`.jpg`, `.png`, `.tif`, etc.)

**Sample Output:**

```
--- Classification Results from folder: test_images/ ---

image1.jpg       --> CLASS 1 (98.52%)
sample_cat.png   --> CLASS 0 (87.44%)
```

---

## 🧠 Model Overview

* **Base Model**: `torchvision.models.vgg16` (pretrained)
* **Classifier**: Modified final layer `nn.Linear(4096, 2)`
* **Feature Extractor**: Frozen (can be customized)
* **Loss Function**: CrossEntropyLoss
* **Optimizer**: Adam (LR = 1e-4)
* **Batch Size**: 32
* **Epochs**: 100 (early stopping enabled)
* **Early Stopping Patience**: 5 epochs

---

## 📊 Outputs

* **Best Model**: `vgg16_best_model.pth` (early stopped)
* **Final Model**: `vgg16_image_classifier.pth` (last epoch)
* **Plot**: `accuracy_plot.png` (training vs validation accuracy)

---

## 📝 Notes

* If using a different number of classes, update:

```python
model.classifier[6] = nn.Linear(4096, <num_classes>)
```

* Ensure your model weights match the classifier architecture.

---


