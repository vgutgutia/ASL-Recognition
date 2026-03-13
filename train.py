"""
ASL Recognition — Training Pipeline
=====================================
Trains a lightweight CNN (<1M params) on ASL letters A-E.
Uses heavy augmentation to compensate for small datasets.
Tracks accuracy, F1, and saves the best model.

Usage:  python3 train.py
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ─────────────────────────────────────────────────────
AUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "augmented")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
DATA_DIR = AUG_DIR if os.path.isdir(AUG_DIR) and len(os.listdir(AUG_DIR)) > 0 else RAW_DIR
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 40
LR = 3e-3
PATIENCE = 12  # early stopping
NUM_CLASSES = 5
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data Augmentation ──────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Model: Lightweight CNN with depthwise separable convs ──────
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
        # Block 1: standard conv
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Block 2: depthwise separable
        self.block2 = nn.Sequential(
            DepthwiseSeparable(64, 128),
            nn.MaxPool2d(2),
        )
        # Block 3: depthwise separable + residual
        self.block3_main = DepthwiseSeparable(128, 256)
        self.block3_skip = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.pool3 = nn.MaxPool2d(2)

        # Block 4: depthwise separable
        self.block4 = nn.Sequential(
            DepthwiseSeparable(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool3(self.block3_main(x) + self.block3_skip(x))  # residual
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Training ───────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def run_evaluation(model, loader, device, class_names=None):
    model.eval()
    all_preds, all_labels = [], []
    running_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return running_loss / total, acc, f1, report, cm, all_preds, all_labels


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history["val_f1"], label="Val F1", color="green")
    axes[2].set_title("Validation F1 Score")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Training plots saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    tick_marks = range(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Confusion matrix saved to {save_path}")


def main():
    print("=" * 60)
    print("  ASL Recognition — Training Pipeline")
    print("=" * 60)
    print(f"  Device    : {DEVICE}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Epochs    : {EPOCHS} (patience={PATIENCE})")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  LR        : {LR}")

    # Load dataset
    full_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    print(f"  Classes   : {class_names}")
    print(f"  Total imgs: {len(full_dataset)}")

    # 80/20 split
    n_val = int(0.2 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_indices, val_indices = random_split(range(len(full_dataset)), [n_train, n_val],
                                              generator=torch.Generator().manual_seed(42))

    # Create datasets with appropriate transforms
    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_subset)} | Val: {len(val_subset)}")

    # Model
    model = ASLNet(num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    if total_params < 1_000_000:
        print(f"  Under 1M params — full 20 competition points!")
    print("=" * 60)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _, _, _, _ = run_evaluation(model, val_loader, DEVICE, class_names)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        lr_now = optimizer.param_groups[0]["lr"]
        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " *best*"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={train_loss:.4f}/{val_loss:.4f}  "
              f"acc={train_acc:.3f}/{val_acc:.3f}  "
              f"f1={val_f1:.3f}  lr={lr_now:.6f}{marker}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # Save best model
    model_path = os.path.join(MODEL_DIR, "asl_model_best.pth")
    torch.save({
        "model_state_dict": best_model_state,
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "best_f1": best_f1,
        "total_params": total_params,
    }, model_path)
    print(f"\n  Best model saved to {model_path}")

    # Final evaluation with best model
    model.load_state_dict(best_model_state)
    val_loss, val_acc, val_f1, report, cm, _, _ = run_evaluation(model, val_loader, DEVICE, class_names)

    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS (best model)")
    print(f"{'=' * 60}")
    print(f"  Accuracy : {val_acc:.4f}")
    print(f"  F1 Score : {val_f1:.4f}")
    print(f"  Params   : {total_params:,}")
    print(f"\n{report}")

    # Plots
    plot_history(history, os.path.join(MODEL_DIR, "training_history.png"))
    plot_confusion_matrix(cm, class_names, os.path.join(MODEL_DIR, "confusion_matrix.png"))

    # Competition score estimate
    print(f"\n{'=' * 60}")
    print(f"  ESTIMATED COMPETITION SCORE")
    print(f"{'=' * 60}")
    acc_pts = val_acc * 100 * 0.20
    f1_pts = val_f1 * 30
    param_pts = 20 if total_params < 1_000_000 else (10 if total_params < 10_000_000 else 0)
    print(f"  Accuracy pts : {acc_pts:.1f} / 20")
    print(f"  F1 pts       : {f1_pts:.1f} / 30")
    print(f"  Param pts    : {param_pts} / 20")
    print(f"  TOTAL        : {acc_pts + f1_pts + param_pts:.1f} / 70")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
