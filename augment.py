# augment.py - offline data augmentation
# takes raw images and creates augmented copies (4 per original)
# run this before training to expand the dataset

import cv2
import os
import numpy as np
import random

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
AUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "augmented")
IMG_SIZE = 300
AUGMENTS_PER_IMAGE = 4


def random_brightness(img):
    factor = random.uniform(0.5, 1.5)
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def random_contrast(img):
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    factor = random.uniform(0.6, 1.4)
    return np.clip(mean + factor * (img - mean), 0, 255).astype(np.uint8)

def random_rotation(img):
    angle = random.uniform(-20, 20)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_shift(img):
    h, w = img.shape[:2]
    tx = random.uniform(-0.15, 0.15) * w
    ty = random.uniform(-0.15, 0.15) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_zoom(img):
    h, w = img.shape[:2]
    scale = random.uniform(0.8, 1.2)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    if scale > 1.0:
        sy = (new_h - h) // 2
        sx = (new_w - w) // 2
        return resized[sy:sy + h, sx:sx + w]
    else:
        py = (h - new_h) // 2
        px = (w - new_w) // 2
        result = np.zeros_like(img)
        result[py:py + new_h, px:px + new_w] = resized
        return result

def random_flip(img):
    if random.random() > 0.5:
        return cv2.flip(img, 1)  # horizontal flip
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
    letters = sorted([d for d in os.listdir(RAW_DIR)
                      if os.path.isdir(os.path.join(RAW_DIR, d)) and d != ".DS_Store"])

    total = 0
    for letter in letters:
        raw_dir = os.path.join(RAW_DIR, letter)
        aug_dir = os.path.join(AUG_DIR, letter)
        os.makedirs(aug_dir, exist_ok=True)

        images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # copy originals
        for name in images:
            img = cv2.imread(os.path.join(raw_dir, name))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(aug_dir, f"orig_{name}"), img)

        # generate augmented versions
        count = 0
        for name in images:
            img = cv2.imread(os.path.join(raw_dir, name))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            for i in range(AUGMENTS_PER_IMAGE):
                aug = augment_image(img)
                cv2.imwrite(os.path.join(aug_dir, f"aug{i}_{name}"), aug)
                count += 1

        letter_total = len(images) + count
        total += letter_total
        print(f"  {letter}: {len(images)} originals + {count} augmented = {letter_total}")

    print(f"\n  Total: {total} images in {AUG_DIR}")


if __name__ == "__main__":
    main()
