ASL Letter Recognition CNN - Report

1. Data Collection Strategy

We built a Python script (collect_data.py) that uses OpenCV to capture images from a webcam. A yellow guide box is drawn on screen, and images are cropped from that box region so the hand fills most of the frame. Each captured image is saved as a 300x300 JPEG into labeled folders (A/, B/, C/, D/, E/).

We collected a total of 890 raw images across multiple sessions:
- A: 220, B: 160, C: 160, D: 160, E: 190

To make the dataset more diverse, we varied lighting (bright, dim, natural), background (plain walls, cluttered desk, dark), used both left and right hands, and tried different angles/distances from the camera. We also captured a set with the hand farther away to make the model more robust.

We then ran offline augmentation (augment.py) to generate 4 variants per image, expanding the dataset to 4,450 images. Each augmented image gets a random combination of: horizontal flip, rotation (up to 20 degrees), translation, zoom, brightness/contrast changes, Gaussian blur, and additive noise.

During training, we also apply lighter on-the-fly augmentation: random flips, rotation, color jitter, and random erasing.

2. Model Architecture

Our model (ASLNet) uses depthwise separable convolutions to keep the parameter count low. Instead of using a regular convolution that processes all channels at once, depthwise separable convs split this into two steps: a 3x3 convolution applied independently to each channel (depthwise), followed by a 1x1 convolution that mixes channels (pointwise). This reduces parameters by roughly 8-9x compared to standard convolutions.

The architecture has 4 blocks:

Block 1: Standard Conv2d (3 -> 64 channels, 3x3 kernel, stride 1, padding 1) + BatchNorm + ReLU + MaxPool 2x2

Block 2: Depthwise Separable Conv (64 -> 128 channels, 3x3 depthwise + 1x1 pointwise, stride 1, padding 1) + MaxPool 2x2

Block 3: Depthwise Separable Conv (128 -> 256 channels, 3x3 depthwise + 1x1 pointwise, stride 1, padding 1) + Residual skip connection (1x1 conv to match dimensions) + MaxPool 2x2

Block 4: Depthwise Separable Conv (256 -> 512 channels, 3x3 depthwise + 1x1 pointwise, stride 1, padding 1) + AdaptiveAvgPool to 1x1

Classifier: Dropout (40%) + Fully Connected layer (512 -> 5 classes)

Block 3 has a residual (skip) connection: the input is passed through both the main depthwise separable path and a 1x1 convolution shortcut, and the outputs are added together. This helps gradients flow during training and improves convergence.

We use BatchNorm after every convolution layer to stabilize training. AdaptiveAvgPool in block 4 replaces the typical flatten operation, which reduces overfitting by collapsing spatial dimensions to 1x1 before the classifier.

Total parameters: 215,557 (well under 1M)

3. Training Approach and Hyperparameters

- Input size: 128x128 RGB
- Batch size: 64
- Optimizer: AdamW (weight decay = 1e-4)
- Learning rate: 3e-3
- LR schedule: Cosine annealing (min lr = 1e-6)
- Loss function: Cross-entropy with label smoothing (0.1)
- Epochs: 40 max (early stopping, patience = 12)
- Train/Val split: 80/20 (seed = 42)

Images are resized to 128x128 and normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). We chose 128x128 over 224x224 because it trains significantly faster with minimal accuracy loss.

Label smoothing (0.1) prevents the model from becoming overconfident on training examples. Cosine annealing gradually reduces the learning rate from 3e-3 down to 1e-6, which helps fine-tune in later epochs.

For evaluation, we use test-time augmentation (TTA): we predict on both the original image and a horizontally flipped version, then average the probabilities. This gives a small accuracy boost for free.

Training was done on Apple Silicon GPU (MPS backend).

4. Results

Best model was saved at epoch 35 based on validation F1 score.

Top-1 Accuracy: 98.3%
Weighted F1: 0.983

Per-class breakdown:
- A: Precision 0.98, Recall 1.00, F1 0.99
- B: Precision 1.00, Recall 0.97, F1 0.98
- C: Precision 0.98, Recall 0.98, F1 0.98
- D: Precision 0.96, Recall 0.98, F1 0.97
- E: Precision 0.99, Recall 0.98, F1 0.99

5. Challenges

A vs E confusion: These two ASL signs look very similar - both are closed fists, differing only in thumb position. Early models struggled here. We fixed it by re-cropping all images to focus tightly on the hand (instead of capturing the full frame), and increasing model capacity from 56k to 216k parameters so it could learn the subtle difference.

Camera selection on macOS: macOS Continuity Camera would sometimes pick an iPhone as the webcam instead of the built-in camera. We added camera scanning and selection to all scripts.

Training speed: Our first attempt at 224x224 resolution was way too slow - 20 minutes for 4 epochs. Dropping to 128x128 with a larger batch size (64) and higher learning rate (3e-3) solved this with no meaningful accuracy loss.

Cropping mismatch: The data collection script was initially saving the full camera frame instead of cropping from the guide box, making hands appear small and far away. Re-cropping all existing images to the box region significantly improved model performance.

Package conflicts: NumPy 2.x had compatibility issues with scikit-learn and other packages in our conda environment. Required downgrading to NumPy 1.26.4 and upgrading several other dependencies.
