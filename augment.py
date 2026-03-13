"""
ASL Data Augmentation — Offline Expansion
==========================================
Takes raw images and generates augmented copies.
Expands each image into 8 variants = ~2400 total from 300 originals.
"""

import cv2
import os
import numpy as np
from pathlib import Path
import random

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
AUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "augmented")
IMG_SIZE = 300
AUGMENTS_PER_IMAGE = 4


def random_brightness(img, low=0.5, high=1.5):
    factor = random.uniform(low, high)
    return np.clip(img * factor, 0, 255).astype(np.uint8)


def random_contrast(img, low=0.6, high=1.4):
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    factor = random.uniform(low, high)
    return np.clip(mean + factor * (img - mean), 0, 255).astype(np.uint8)


def random_rotation(img, max_angle=20):
    angle = random.uniform(-max_angle, max_angle)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_shift(img, max_shift=0.15):
    h, w = img.shape[:2]
    tx = random.uniform(-max_shift, max_shift) * w
    ty = random.uniform(-max_shift, max_shift) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_zoom(img, low=0.8, high=1.2):
    h, w = img.shape[:2]
    scale = random.uniform(low, high)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    if scale > 1.0:
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return resized[start_y:start_y + h, start_x:start_x + w]
    else:
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        result = np.zeros_like(img)
        result[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return result


def random_flip(img):
    if random.random() > 0.5:
        return cv2.flip(img, 1)
    return img


def random_blur(img):
    if random.random() > 0.5:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img


def random_noise(img):
    if random.random() > 0.6:
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def augment_image(img):
    """Apply a random combination of augmentations."""
    img = random_flip(img)
    img = random_rotation(img)
    img = random_shift(img)
    img = random_zoom(img)
    img = random_brightness(img)
    img = random_contrast(img)
    img = random_blur(img)
    img = random_noise(img)
    return img


def main():
    print("=" * 60)
    print("  ASL Offline Augmentation")
    print("=" * 60)

    letters = sorted([d for d in os.listdir(RAW_DIR)
                      if os.path.isdir(os.path.join(RAW_DIR, d)) and d != ".DS_Store"])

    total_generated = 0

    for letter in letters:
        raw_letter_dir = os.path.join(RAW_DIR, letter)
        aug_letter_dir = os.path.join(AUG_DIR, letter)
        os.makedirs(aug_letter_dir, exist_ok=True)

        images = [f for f in os.listdir(raw_letter_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Copy originals first
        for img_name in images:
            img = cv2.imread(os.path.join(raw_letter_dir, img_name))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(aug_letter_dir, f"orig_{img_name}"), img)

        # Generate augmented copies
        count = 0
        for img_name in images:
            img = cv2.imread(os.path.join(raw_letter_dir, img_name))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            for i in range(AUGMENTS_PER_IMAGE):
                aug = augment_image(img)
                out_name = f"aug{i}_{img_name}"
                cv2.imwrite(os.path.join(aug_letter_dir, out_name), aug)
                count += 1

        total_letter = len(images) + count
        total_generated += total_letter
        print(f"  {letter}: {len(images)} originals + {count} augmented = {total_letter}")

    print(f"\n  Total: {total_generated} images in {AUG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
