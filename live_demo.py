"""
ASL Live Detection Demo
========================
Opens your webcam and shows real-time predictions
with a confidence bar for each letter.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import os

# ── Import model class from train.py ──────────────────────────
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
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

COLORS = {
    "A": (0, 200, 255),
    "B": (255, 100, 100),
    "C": (100, 255, 100),
    "D": (100, 100, 255),
    "E": (255, 200, 100),
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def draw_confidence_bars(frame, probs, class_names, x=10, y_start=80, bar_w=200, bar_h=30):
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        y = y_start + i * (bar_h + 12)
        color = COLORS.get(name, (200, 200, 200))
        fill = int(bar_w * prob)

        cv2.putText(frame, name, (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        bx = x + 35
        cv2.rectangle(frame, (bx, y), (bx + bar_w, y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, y), (bx + fill, y + bar_h), color, -1)
        cv2.rectangle(frame, (bx, y), (bx + bar_w, y + bar_h), (200, 200, 200), 1)

        cv2.putText(frame, f"{prob * 100:.1f}%", (bx + bar_w + 10, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]

    model = ASLNet(num_classes=checkpoint["num_classes"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Model loaded ({checkpoint['total_params']:,} params, best F1={checkpoint['best_f1']:.3f})")

    # Camera selection
    print("  Scanning for cameras...")
    available = []
    for idx in range(5):
        test = cv2.VideoCapture(idx)
        if test.isOpened():
            ret, frame = test.read()
            if ret:
                h, w = frame.shape[:2]
                available.append((idx, f"Camera {idx}  ({w}x{h})"))
            test.release()

    if not available:
        print("ERROR: No cameras found.")
        return

    if len(available) == 1:
        cam_idx = available[0][0]
    else:
        print(f"  Found {len(available)} cameras:")
        for idx, desc in available:
            print(f"    [{idx}] {desc}")
        while True:
            try:
                choice = int(input("  Pick a camera number: "))
                if choice in [a[0] for a in available]:
                    cam_idx = choice
                    break
            except ValueError:
                pass

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("  Live demo running. Press Q to quit.")

    history = []
    SMOOTH_N = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        box_size = 250
        half = box_size // 2
        bx1, by1 = cx - half, cy - half
        bx2, by2 = cx + half, cy + half

        cropped = frame[by1:by2, bx1:bx2]
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        input_tensor = transform(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        history.append(probs)
        if len(history) > SMOOTH_N:
            history.pop(0)
        avg_probs = np.mean(history, axis=0)

        predicted_idx = np.argmax(avg_probs)
        predicted_letter = class_names[predicted_idx]
        confidence = avg_probs[predicted_idx]

        color = COLORS.get(predicted_letter, (255, 255, 255))
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)

        cv2.putText(frame, predicted_letter, (w - 120, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5)
        cv2.putText(frame, f"{confidence * 100:.0f}%", (w - 120, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        draw_confidence_bars(frame, avg_probs, class_names)

        cv2.putText(frame, "Place hand in box | Q to quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("ASL Live Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
