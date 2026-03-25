# train_v2.py - fine-tune ViT for ASL recognition
# uses background pasting to simulate messy real-world images

import os
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import Counter
from torchvision import models

# config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINED_DIR = os.path.join(BASE_DIR, "data", "combined")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
BG_DIR = os.path.join(BASE_DIR, "data", "backgrounds")
DATA_DIR = COMBINED_DIR if os.path.isdir(COMBINED_DIR) else RAW_DIR
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
PATIENCE = 10
NUM_CLASSES = 5
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

os.makedirs(MODEL_DIR, exist_ok=True)


class PasteOnBackground:
    """paste hand crop onto a random background at random position/scale"""
    def __init__(self, bg_dir, prob=0.4):
        self.bg_paths = []
        if os.path.isdir(bg_dir):
            self.bg_paths = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir)
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.prob = prob

    def __call__(self, img):
        if random.random() > self.prob or not self.bg_paths:
            return img
        try:
            bg = Image.open(random.choice(self.bg_paths)).convert("RGB").resize((448, 448))
            scale = random.uniform(0.2, 0.7)
            hand_size = int(448 * scale)
            hand_resized = img.resize((hand_size, hand_size))
            max_x = max(0, 448 - hand_size)
            max_y = max(0, 448 - hand_size)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            bg.paste(hand_resized, (x, y))
            return bg
        except Exception:
            return img


# NO horizontal flip - flipping changes ASL letter meaning
train_transform = transforms.Compose([
    PasteOnBackground(BG_DIR, prob=0.4),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.5, hue=0.15),
    transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.6, 1.3)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.RandomGrayscale(p=0.15),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.25)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model():
    # ViT-B/16 from torchvision - self-attention focuses on hand in cluttered images
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    # freeze early encoder blocks, train last 4 + head
    for name, param in model.named_parameters():
        if "encoder.layers.encoder_layer_" in name:
            block_num = int(name.split("encoder_layer_")[1].split(".")[0])
            if block_num < 8:
                param.requires_grad = False
        elif "conv_proj" in name or "class_token" in name or "encoder.pos_embedding" in name:
            param.requires_grad = False
    # replace head (vit_b_16 has 768 features)
    model.heads = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(768, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


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


def main():
    print(f"Training on {DEVICE}, img_size={IMG_SIZE}, batch={BATCH_SIZE}, lr={LR}")
    print(f"Data: {DATA_DIR}")
    print(f"Model: ViT-Small (timm), fine-tuning blocks 8-11 + head")

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

    train_labels = [full_dataset.targets[i] for i in train_idx.indices]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")

    model = build_model().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # separate lr for pretrained blocks vs head
    head_params = [p for n, p in model.named_parameters() if "heads" in n]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "heads" not in n]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR * 0.5},
        {"params": head_params, "lr": LR * 5},
    ], weight_decay=1e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

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

    model_path = os.path.join(MODEL_DIR, "asl_model_best.pth")
    torch.save({
        "model_state_dict": best_state,
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "best_f1": best_f1,
        "total_params": total_params,
        "arch": "vit_small",
    }, model_path)
    print(f"\n  Saved to {model_path}")

    model.load_state_dict(best_state)
    val_loss, val_acc, val_f1, report, cm = validate(model, val_loader, class_names)
    print(f"\n  Final: acc={val_acc:.4f}, f1={val_f1:.4f}")
    print(report)

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep_range = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep_range, history["train_loss"], label="Train")
    axes[0].plot(ep_range, history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(ep_range, history["train_acc"], label="Train")
    axes[1].plot(ep_range, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    axes[2].plot(ep_range, history["val_f1"], label="F1", color="green")
    axes[2].set_title("F1"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_history.png"), dpi=150)
    print("  Saved plots")


if __name__ == "__main__":
    main()
