"""
ASL Data Collection — 30 per letter
=====================================
150 total. Press SPACE to capture, S to skip letter, Q to quit.
"""

import cv2
import os
from datetime import datetime

LETTERS = ["A", "B", "C", "D", "E"]
IMAGES_PER_LETTER = 15
BOX_SIZE = 250
IMG_SIZE = 300
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")


def overlay_text(frame, lines, start_y=30, color=(0, 255, 0), scale=0.7, thickness=2):
    for i, line in enumerate(lines):
        y = start_y + i * 35
        cv2.putText(frame, line, (12, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def get_box_coords(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = BOX_SIZE // 2
    return cx - half, cy - half, cx + half, cy + half


def draw_center_box(frame):
    x1, y1, x2, y2 = get_box_coords(frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


def main():
    for letter in LETTERS:
        os.makedirs(os.path.join(DATA_DIR, letter), exist_ok=True)

    # Camera selection
    print("\n  Scanning for cameras...")
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
        print(f"\n  Found {len(available)} cameras:")
        for idx, desc in available:
            print(f"    [{idx}] {desc}")
        while True:
            try:
                choice = int(input("\n  Pick a camera number: "))
                if choice in [a[0] for a in available]:
                    cam_idx = choice
                    break
            except ValueError:
                pass

    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    total = 0
    quit_early = False

    for letter in LETTERS:
        if quit_early:
            break

        captured = 0
        flash_timer = 0

        # "Get ready" screen
        for _ in range(90):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
            overlay_text(frame, [
                f"GET READY — Letter {letter}",
                f"",
                f"Show the ASL sign for '{letter}' with your hand",
                f"Take {IMAGES_PER_LETTER} photos — vary lighting, hand, background",
                f"Press SPACE to start",
            ], start_y=80, color=(0, 200, 255), scale=0.75)
            cv2.imshow("ASL Data Collection", frame)
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                quit_early = True
                break
            if key == ord(' ') or key == ord('s'):
                break

        if quit_early:
            break

        # Capture loop
        while captured < IMAGES_PER_LETTER:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            draw_center_box(display)

            overlay_text(display, [
                f"Letter {letter}  —  {captured}/{IMAGES_PER_LETTER}    [SPACE=capture  S=skip  Q=quit]",
            ], start_y=25, scale=0.55, thickness=1)

            # Progress bar
            bar_y = display.shape[0] - 40
            cv2.rectangle(display, (10, bar_y), (310, bar_y + 20), (255, 255, 255), 2)
            fill = int(296 * captured / IMAGES_PER_LETTER)
            cv2.rectangle(display, (12, bar_y + 2), (12 + fill, bar_y + 18), (0, 255, 0), -1)

            if flash_timer > 0:
                alpha = flash_timer / 8.0
                white = display.copy()
                white[:] = (255, 255, 255)
                display = cv2.addWeighted(white, alpha * 0.3, display, 1 - alpha * 0.3, 0)
                flash_timer -= 1

            cv2.imshow("ASL Data Collection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                x1, y1, x2, y2 = get_box_coords(frame)
                cropped = frame[y1:y2, x1:x2]
                resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{letter}_{timestamp}.jpg"
                cv2.imwrite(os.path.join(DATA_DIR, letter, filename), resized)

                captured += 1
                total += 1
                flash_timer = 6

            elif key == ord('s'):
                break
            elif key == ord('q'):
                quit_early = True
                break

        print(f"  {letter}: {captured} images captured")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Done! {total} total images saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
