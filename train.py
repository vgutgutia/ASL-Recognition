# train.py - trains the ASL recognition CNN
# uses augmented data if available, otherwise raw
# saves best model based on validation F1 score

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

# config
AUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "augmented")
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
DATA_DIR = AUG_DIR if os.path.isdir(AUG_DIR) and len(os.listdir(AUG_DIR)) > 0 else RAW_DIR
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 40
LR = 3e-3
PATIENCE = 12
NUM_CLASSES = 5
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# augmentation for training - keep it light since we already augmented offline
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# depthwise separable conv - way fewer params than a regular conv
# basically splits it into a per-channel 3x3 conv then a 1x1 mixing conv
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
        # block 1: regular conv, 3 -> 64 channels, 3x3 kernel, stride 1, padding 1
        # followed by batchnorm + relu + 2x2 maxpool
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # block 2: depthwise separable, 64 -> 128
        # depthwise: 3x3 kernel per channel, pointwise: 1x1 to mix
        # then 2x2 maxpool
        self.block2 = nn.Sequential(
            DepthwiseSeparable(64, 128),
            nn.MaxPool2d(2),
        )
        # block 3: depthwise separable 128 -> 256 with a residual skip connection
        # the skip uses a 1x1 conv to match dimensions (128 -> 256)
        # then 2x2 maxpool
        self.block3_main = DepthwiseSeparable(128, 256)
        self.block3_skip = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.pool3 = nn.MaxPool2d(2)

        # block 4: depthwise separable 256 -> 512
        # then adaptive avg pool to 1x1 (replaces flattening, less overfitting)
        self.block4 = nn.Sequential(
            DepthwiseSeparable(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        # classifier: dropout then one fully connected layer, 512 -> 5 classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)       # 128x128 -> 64x64
        x = self.block2(x)       # 64x64 -> 32x32
        # residual connection: add skip to main path
        x = self.pool3(self.block3_main(x) + self.block3_skip(x))  # 32x32 -> 16x16
        x = self.block4(x)       # 16x16 -> 1x1
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def validate(model, loader, class_names=None):
    model.eval()
    all_preds, all_labels = [], []
    running_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
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
    return running_loss / total, acc, f1, report, cm


def save_plots(history, cm, class_names):
    # training curves
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
    axes[2].set_title("F1 Score")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_history.png"), dpi=150)

    # confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)


def main():
    print(f"Training on {DEVICE}, img_size={IMG_SIZE}, batch={BATCH_SIZE}, lr={LR}")
    print(f"Data: {DATA_DIR}")

    # load and split dataset
    full_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}, Total: {len(full_dataset)} images")

    n_val = int(0.2 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_idx, val_idx = random_split(range(len(full_dataset)), [n_train, n_val],
                                       generator=torch.Generator().manual_seed(42))

    train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    train_subset = torch.utils.data.Subset(train_dataset, train_idx.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx.indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")

    # setup model
    model = ASLNet(num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, class_names)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        lr = optimizer.param_groups[0]["lr"]
        tag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            tag = " *best*"
        else:
            no_improve += 1

        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={train_loss:.4f}/{val_loss:.4f}  "
              f"acc={train_acc:.3f}/{val_acc:.3f}  "
              f"f1={val_f1:.3f}  lr={lr:.6f}{tag}")

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # save best model
    model_path = os.path.join(MODEL_DIR, "asl_model_best.pth")
    torch.save({
        "model_state_dict": best_state,
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "best_f1": best_f1,
        "total_params": total_params,
    }, model_path)
    print(f"\n  Saved best model to {model_path}")

    # final eval
    model.load_state_dict(best_state)
    val_loss, val_acc, val_f1, report, cm = validate(model, val_loader, class_names)
    print(f"\n  Final: acc={val_acc:.4f}, f1={val_f1:.4f}, params={total_params:,}")
    print(report)

    save_plots(history, cm, class_names)
    print("  Saved training plots and confusion matrix")


if __name__ == "__main__":
    main()
