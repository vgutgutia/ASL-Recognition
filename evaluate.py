"""
ASL Recognition — Evaluation on Secret Test Dataset
=====================================================
Loads the trained model and evaluates on the teacher's test data.

Usage:  python3 evaluate.py <path_to_test_folder>

The test folder should have subfolders: A/ B/ C/ D/ E/
each containing 300x300 images.
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Model Definition ───────────────────────────────────────────
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class ASLNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            DepthwiseSeparable(64, 128),
            nn.MaxPool2d(2),
        )
        self.block3_main = DepthwiseSeparable(128, 256)
        self.block3_skip = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.pool3 = nn.MaxPool2d(2)
        self.block4 = nn.Sequential(
            DepthwiseSeparable(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool3(self.block3_main(x) + self.block3_skip(x))
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Config ─────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "asl_model_best.pth")
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate.py <path_to_test_folder>")
        print("  The test folder should contain subfolders: A/ B/ C/ D/ E/")
        return

    test_dir = sys.argv[1]
    if not os.path.isdir(test_dir):
        print(f"ERROR: {test_dir} is not a valid directory")
        return

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = ASLNet(num_classes=checkpoint["num_classes"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 60)
    print("  ASL Recognition — Secret Dataset Evaluation")
    print("=" * 60)
    print(f"  Model     : {MODEL_PATH}")
    print(f"  Params    : {checkpoint['total_params']:,}")
    print(f"  Train F1  : {checkpoint['best_f1']:.4f}")
    print(f"  Device    : {DEVICE}")
    print(f"  Test dir  : {test_dir}")
    print(f"  Img size  : {img_size}x{img_size}")
    print(f"  Classes   : {class_names}")

    # Preprocessing — matches training val_transform
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"  Images    : {len(test_dataset)}")
    print(f"  Detected  : {test_dataset.classes}")
    print("=" * 60)

    # Run inference
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds,
                                   target_names=test_dataset.classes, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Results
    print(f"\n{'=' * 60}")
    print(f"  RESULTS ON SECRET DATASET")
    print(f"{'=' * 60}")
    print(f"  Top-1 Accuracy : {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(f"  F1 Score       : {f1:.4f}")
    print(f"  Parameters     : {checkpoint['total_params']:,}")
    print(f"\n{report}")

    # Competition scoring
    acc_pts = accuracy * 100 * 0.20
    f1_pts = f1 * 30
    param_pts = 20 if checkpoint['total_params'] < 1_000_000 else (10 if checkpoint['total_params'] < 10_000_000 else 0)

    print(f"{'=' * 60}")
    print(f"  COMPETITION SCORE")
    print(f"{'=' * 60}")
    print(f"  Accuracy pts  : {acc_pts:.1f} / 20")
    print(f"  F1 pts        : {f1_pts:.1f} / 30")
    print(f"  Param pts     : {param_pts} / 20")
    print(f"  TOTAL         : {acc_pts + f1_pts + param_pts:.1f} / 70")
    print(f"{'=' * 60}")

    # Save confusion matrix
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix — Secret Dataset")
    plt.colorbar(im, ax=ax)
    tick_marks = range(len(test_dataset.classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(test_dataset.classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(test_dataset.classes)
    for i in range(len(test_dataset.classes)):
        for j in range(len(test_dataset.classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix_test.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n  Confusion matrix saved to {save_path}")


if __name__ == "__main__":
    main()
