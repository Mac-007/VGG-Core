# Testing Script for VGG16 Image Classifier

This script performs image classification using a trained VGG16 model on a folder of test images. It loads model weights, runs inference on each image in the specified directory, and outputs the predicted class along with confidence scores.

**Author**: `Dr. Amit Chougule, PhD`
**Date**: 25/07/2025

---

## 🔍 Features

- Loads a saved VGG16 model (`.pth` format)
- Performs image preprocessing (resizing, normalization)
- Classifies each image in the given folder
- Outputs class prediction and confidence score for each image
- Compatible with GPU (CUDA) or CPU

---

## 🗂️ Directory Structure

```

project/
│
├── testing/
│   └── test.py               # Inference script
├── vgg16_best_model.pth      # Example trained model file
├── test_images/              # Folder containing images for testing

```

---

## 📦 Requirements

Install the required Python libraries:

```bash
pip install torch torchvision pillow
````

---

## 🖼️ Test Image Format

Ensure your test images are placed inside a folder (e.g., `test_images/`) and are of common image types: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`.

---

## 🚀 How to Run

Run the script from the terminal with the following command:

```bash
python test.py --model_path vgg16_best_model.pth --image_folder test_images/
```

* `--model_path`: Path to the `.pth` file containing trained VGG16 weights.
* `--image_folder`: Path to a folder containing the test images.

---

## 🧠 Model Assumptions

* **Base Model**: VGG16 from `torchvision.models`
* **Classifier Output**: Assumes 2 output classes. Update `model.classifier[6]` if your model uses a different number of classes.
* **Model Weights**: Must match the architecture (especially the number of output classes).

---

## 📝 Output Example

```
--- Classification Results from folder: test_images/ ---

image1.jpg        --> CLASS 1 (98.52%)
sample_cat.png    --> CLASS 0 (87.44%)

```

Each image is classified, and its predicted class index and confidence score are printed.

---

## ⚠️ Notes

* If your trained model is for more than 2 classes, adjust this line in `test.py`:

  ```python
  model.classifier[6] = torch.nn.Linear(4096, <num_classes>)
  ```

* Make sure that the model weights file corresponds to this modified architecture.

---
