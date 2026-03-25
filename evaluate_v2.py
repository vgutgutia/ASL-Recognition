# evaluate_v2.py - simple eval with flip TTA
# usage: python3 evaluate_v2.py <path_to_test_folder>

import sys
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, classification_report

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "asl_model_best.pth")
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
NUM_CLASSES = 5


def build_model(num_classes=5):
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1536, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate_v2.py <path_to_test_folder>")
        return

    test_dir = sys.argv[1]
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = build_model(num_classes=checkpoint["num_classes"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model: EfficientNet-B3, train F1={checkpoint['best_f1']:.4f}")

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    print(f"Test images: {len(test_dataset)}")

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in test_dataset:
            tensor = img.unsqueeze(0).to(DEVICE)
            probs = torch.softmax(model(tensor), dim=1)
            probs += torch.softmax(model(torch.flip(tensor, dims=[3])), dim=1)
            pred = probs.squeeze().argmax().item()
            if pred == label:
                correct += 1
            total += 1
            all_preds.append(pred)
            all_labels.append(label)

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds,
                                   target_names=test_dataset.classes, zero_division=0)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"F1 Score: {f1:.4f}")
    print(report)


if __name__ == "__main__":
    main()
