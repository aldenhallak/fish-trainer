# fish-trainer

A PyTorch-based CNN project for classifying fish doodles vs. not-fish doodles, using ResNet18 and QuickDraw data. Includes robust preprocessing for transparency, early stopping, ONNX export, and test scripts.

This was made in conjunction with my [fishes](https://github.com/aldenhallak/fishes) project.

## Repository Structure

- `train_fish_doodle_classifier.py`  
  Main training script. Handles dataset loading, preprocessing, model training (with early stopping), and ONNX export. Also includes utilities for generating datasets from Google QuickDraw.

- `test_fish_classifier.py`  
  Script for evaluating the trained model (PyTorch or ONNX) on your dataset or external images. Saves transformed images for inspection.

- `requirements.txt`  
  Python dependencies for training and testing (PyTorch, torchvision, onnx, onnxruntime, scikit-learn, tqdm, Pillow, numpy, opencv-python).

- `dataset/`  
  Directory for your training images, with subfolders `fish/` and `not_fish/`.

- `quickdraw/`  
  Contains downloaded QuickDraw `.ndjson` files for generating synthetic training data.

- `fish_doodle_classifier.pth`  
  Saved PyTorch model weights after training.

- `fish_doodle_classifier.onnx`  
  Exported ONNX model for cross-platform inference.

- `test_fish_classifier.py`  
  Script to test the model on your dataset or custom images, using either PyTorch or ONNX.

- `README.md`  
  Here :)

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset:**
   - Place fish images in `dataset/fish/` and not-fish images in `dataset/not_fish/`.
   - Or, use QuickDraw data by running:
     ```bash
     python train_fish_doodle_classifier.py --pretrain
     ```

3. **Train the model:**
   ```bash
   python train_fish_doodle_classifier.py
   ```
   - Early stopping is enabled to prevent overfitting.
   - The best model is exported as both `.pth` and `.onnx`.

4. **Test the model:**
   ```bash
   python test_fish_classifier.py
   ```
   - Evaluates on your dataset and prints classification metrics.
   - Also supports ONNX inference for deployment.

## Key Design Decisions

- **Transparency Handling:**
  All images are composited onto a white background before preprocessing. So if your fish is all white, it won't work.

- **Early Stopping:**
  Training halts if validation loss does not improve for 5 epochs, reducing overfitting.

- **ONNX Export:**
  Model is exported to ONNX for compatibility with non-PyTorch environments. I use this in the [fishes frontend](https://github.com/aldenhallak/fishes).

- **QuickDraw Integration:**
  Scripts can auto-download and convert QuickDraw doodles for both fish and not-fish classes. I ended up not using this, but have left it in the repo in case someone else wants to.

- **Consistent Preprocessing:**
  The same preprocessing pipeline is used for both training and inference, including in the test script.

- **Class Imbalance:**
  Weighted sampling and loss are used to address class imbalance between fish and not-fish.

## Notes
- The model expects 224x224 grayscale images (3 channels for ResNet compatibility).
- All code is designed for clarity and reproducibility.
- For best results, inspect the saved transformed images to verify preprocessing.

---

Feel free to modify the scripts for your own dataset or use case!
