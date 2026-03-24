# evaluate_v2.py - evaluate the pretrained ResNet18 model on test data
# usage: python3 evaluate_v2.py <path_to_test_folder>

import sys
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "asl_model_best.pth")
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def build_model(num_classes=5):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate_v2.py <path_to_test_folder>")
        return

    test_dir = sys.argv[1]
    if not os.path.isdir(test_dir):
        print(f"Error: {test_dir} not found")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = build_model(num_classes=checkpoint["num_classes"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model: ResNet18, {checkpoint['total_params']:,} params, train F1={checkpoint['best_f1']:.4f}")
    print(f"Device: {DEVICE}, img size: {img_size}x{img_size}")

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test images: {len(test_dataset)}, classes: {test_dataset.classes}")

    # TTA: original + flipped
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            probs_orig = torch.softmax(model(images), dim=1)
            probs_flip = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            probs = (probs_orig + probs_flip) / 2
            preds = probs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds,
                                   target_names=test_dataset.classes, zero_division=0)

    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"F1 Score: {f1:.4f}")
    print(report)


if __name__ == "__main__":
    main()
