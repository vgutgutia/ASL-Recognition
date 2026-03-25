# evaluate_v2.py - evaluate with hand detection preprocessing
# uses MediaPipe to find and crop the hand, then classifies the crop
# usage: python3 evaluate_v2.py <path_to_test_folder>

import sys
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report
from PIL import Image
import mediapipe as mp

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


def detect_hand_crop(img_bgr, hands_detector, padding=0.3):
    """use mediapipe to find the hand and return a cropped image"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    results = hands_detector.process(img_rgb)

    if results.multi_hand_landmarks:
        # get bounding box from landmarks
        hand = results.multi_hand_landmarks[0]
        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # add padding
        bw = x_max - x_min
        bh = y_max - y_min
        x_min = max(0, x_min - bw * padding)
        x_max = min(1, x_max + bw * padding)
        y_min = max(0, y_min - bh * padding)
        y_max = min(1, y_max + bh * padding)

        # make it square
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        side = max(x_max - x_min, y_max - y_min)
        x_min = max(0, cx - side / 2)
        x_max = min(1, cx + side / 2)
        y_min = max(0, cy - side / 2)
        y_max = min(1, cy + side / 2)

        crop = img_rgb[int(y_min * h):int(y_max * h), int(x_min * w):int(x_max * w)]
        if crop.size > 0:
            return Image.fromarray(crop)

    # fallback: return full image
    return Image.fromarray(img_rgb)


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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # init mediapipe hand detector
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,  # low threshold to catch more hands
    )

    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    detected = 0

    # walk through test folders
    class_dirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    label_map = {name: i for i, name in enumerate(class_dirs)}

    for class_name in class_dirs:
        class_dir = os.path.join(test_dir, class_name)
        label = label_map[class_name]

        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                continue

            # detect and crop hand
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            hand_found = results.multi_hand_landmarks is not None

            if hand_found:
                detected += 1
                crop = detect_hand_crop(img_bgr, hands, padding=0.3)
            else:
                crop = Image.fromarray(img_rgb)

            # TTA: original + flipped, and if no hand detected also try center crop
            tensor = to_tensor(crop).unsqueeze(0).to(DEVICE)
            flipped = torch.flip(tensor, dims=[3])

            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)
                probs += torch.softmax(model(flipped), dim=1)

                if not hand_found:
                    # try center crop too since hand might be in the middle
                    center_t = transforms.Compose([
                        transforms.Resize((280, 280)),
                        transforms.CenterCrop(img_size),
                        transforms.ToTensor(),
                        normalize,
                    ])
                    center = center_t(crop).unsqueeze(0).to(DEVICE)
                    probs += torch.softmax(model(center), dim=1)
                    probs += torch.softmax(model(torch.flip(center, dims=[3])), dim=1)

            pred = probs.squeeze().argmax().item()
            if pred == label:
                correct += 1
            total += 1
            all_preds.append(pred)
            all_labels.append(label)

            status = "OK" if pred == label else "WRONG"
            pred_name = class_dirs[pred]
            print(f"  {class_name}/{fname}: pred={pred_name} {'(hand found)' if hand_found else '(no hand)'} {status}")

    hands.close()

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds,
                                   target_names=class_dirs, zero_division=0)

    print(f"\nHands detected: {detected}/{total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"F1 Score: {f1:.4f}")
    print(report)


if __name__ == "__main__":
    main()
