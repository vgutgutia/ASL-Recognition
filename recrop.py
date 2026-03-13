"""
Re-crop existing images to simulate the 250px box crop.
Original images were 300x300 from a full-frame center crop.
This re-crops the center 250/480 ratio (~52%) of each image
to approximate what the box would have captured.
"""

import cv2
import os

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
OUTPUT_SIZE = 300
# The box was 250px in a 480px tall frame = 52% of frame height
# Original images captured the full 480px height cropped to square
# So we need the center 52% of each 300x300 image
CROP_RATIO = 250 / 480

count = 0
for letter in sorted(os.listdir(RAW_DIR)):
    letter_dir = os.path.join(RAW_DIR, letter)
    if not os.path.isdir(letter_dir) or letter.startswith('.'):
        continue
    for fname in os.listdir(letter_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        fpath = os.path.join(letter_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        h, w = img.shape[:2]
        crop_size = int(min(h, w) * CROP_RATIO)
        cx, cy = w // 2, h // 2
        half = crop_size // 2
        cropped = img[cy - half:cy + half, cx - half:cx + half]
        resized = cv2.resize(cropped, (OUTPUT_SIZE, OUTPUT_SIZE))
        cv2.imwrite(fpath, resized)
        count += 1

print(f"Re-cropped {count} images")
