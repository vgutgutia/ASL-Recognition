# ASL Letter Recognition CNN — Project Report

## 1. Data Collection Strategy

We collected images using a Python script with OpenCV that captures frames from a webcam. Images are cropped from a 250x250 pixel guide box centered on the screen, ensuring the hand fills most of the frame. Each image is saved as a 300x300 JPEG into labeled folders (`A/`, `B/`, `C/`, `D/`, `E/`).

**Raw dataset**: 840 images total (A: 210, B: 150, C: 150, D: 150, E: 180)

To ensure diversity, we varied:
- **Lighting**: bright, dim, and natural light
- **Background**: plain walls, cluttered desks, dark backgrounds
- **Hand**: both left and right hands
- **Orientation**: different angles and distances from the camera

We then applied offline augmentation (4 variants per image) to expand the dataset to **4,200 images**. Augmentations include random horizontal flips, rotation (up to 20 degrees), translation, zoom, brightness/contrast changes, Gaussian blur, and additive noise.

During training, additional on-the-fly augmentations are applied: random perspective warping, color jitter, Gaussian blur, and random erasing.

## 2. Model Architecture

We designed a lightweight CNN using **depthwise separable convolutions** to keep the parameter count low while maintaining strong feature extraction. The architecture includes a residual (skip) connection in Block 3.

| Layer | Type | Input Channels | Output Channels | Kernel | Stride | Padding | Pooling |
|-------|------|---------------|-----------------|--------|--------|---------|---------|
| Block 1 | Standard Conv2d + BatchNorm + ReLU | 3 (RGB) | 64 | 3x3 | 1 | 1 | MaxPool 2x2 |
| Block 2 | Depthwise Separable Conv | 64 | 128 | 3x3 (dw) + 1x1 (pw) | 1 | 1 | MaxPool 2x2 |
| Block 3 | Depthwise Separable Conv + Residual Skip | 128 | 256 | 3x3 (dw) + 1x1 (pw) | 1 | 1 | MaxPool 2x2 |
| Block 4 | Depthwise Separable Conv | 256 | 512 | 3x3 (dw) + 1x1 (pw) | 1 | 1 | AdaptiveAvgPool to 1x1 |
| Classifier | Dropout (0.4) + Fully Connected | 512 | 5 | — | — | — | — |

**Design choices**:
- **Depthwise separable convolutions**: Each conv is split into a depthwise 3x3 convolution (processes each channel independently) followed by a 1x1 pointwise convolution (mixes channels). This reduces parameters by ~8-9x compared to standard convolutions.
- **Residual connection in Block 3**: The skip connection (1x1 conv from 128 to 256 channels) helps gradients flow during training, improving convergence.
- **BatchNorm after every conv layer**: Stabilizes training and allows higher learning rates.
- **AdaptiveAvgPool**: Replaces flattening with global average pooling, reducing overfitting.
- **Dropout (40%)**: Applied before the final linear layer to prevent overfitting on our relatively small dataset.

**Total parameters**: 215,557 (well under the 1M threshold for full competition points)

## 3. Training Approach and Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Input size | 128x128 RGB |
| Batch size | 64 |
| Optimizer | AdamW (weight decay = 1e-4) |
| Learning rate | 3e-3 |
| LR schedule | Cosine annealing (eta_min = 1e-6) |
| Loss function | Cross-entropy with label smoothing (0.1) |
| Epochs | 40 (with early stopping, patience = 12) |
| Train/Val split | 80% / 20% (seed = 42) |

**Preprocessing pipeline**:
1. Resize all images to 128x128
2. Convert to tensor (pixel values 0-1)
3. Normalize with ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])

Training was performed on Apple Silicon (M-series) GPU via PyTorch MPS backend.

## 4. Results

**Validation set performance (best model at epoch 39)**:

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 96.8% |
| Weighted F1 Score | 0.968 |

**Per-class performance**:

| Letter | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| A | 0.99 | 0.98 | 0.98 |
| B | 0.94 | 0.97 | 0.96 |
| C | 0.95 | 0.99 | 0.97 |
| D | 0.95 | 0.96 | 0.95 |
| E | 0.99 | 0.95 | 0.97 |

**Estimated competition score**: 68.4 / 70

## 5. Challenges Faced

- **A vs. E confusion**: These two ASL signs are very similar (both closed fists, differing only in thumb position). We solved this by re-cropping all images to focus tightly on the hand and increasing model capacity from 56k to 216k parameters so the model could learn the subtle thumb difference.
- **Camera selection**: macOS Continuity Camera would sometimes select an iPhone as the webcam instead of the built-in camera. We added a camera selection prompt to all scripts.
- **NumPy version conflicts**: The Anaconda environment had NumPy 2.x incompatibilities with scikit-learn, matplotlib, and other packages. Required upgrading several dependencies.
- **Training speed**: Initial runs at 224x224 resolution were too slow. We found 128x128 with a larger batch size (64) and higher learning rate (3e-3) gave faster convergence without sacrificing accuracy.
- **Cropping mismatch**: The data collection script was initially cropping the full camera frame rather than the guide box region, making hands appear small. Re-cropping all images to the box region significantly improved model performance.
