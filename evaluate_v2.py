# evaluate_v2.py - evaluate ViT model with TTA
# usage: python3 evaluate_v2.py <path_to_test_folder>

import sys
import os
import torch
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, classification_report
import timm

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "asl_model_best.pth")
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate_v2.py <path_to_test_folder>")
        return

    test_dir = sys.argv[1]
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Model: ViT-Small, train F1={checkpoint['best_f1']:.4f}")

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
