# Copyright (c) 2025 YI-AN YEH
# This project is licensed under the MIT License - see the LICENSE file for details.

# visualize_augmentations.py

"""
A script to visualize the effects of individual data augmentation techniques
on a single sample image, with precise English labeling for each transformation.
"""

import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. 設定參數 ---
IMAGE_PATH = r"C:\Code\ClassHomeWork\plantdisease\PlantVillage\Tomato_Septoria_leaf_spot\5bf5bfc0-8efe-4f29-814d-c6e594af2ba2___Matt.S_CG 7844.JPG"
IMG_SIZE = 224
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(OUTPUT_DIR, 'augmentation_examples_detailed_en.png') # 新增 _en 以區分

# --- 2. 分別定義每一種資料增強技術 ---
torch.manual_seed(42)

horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
random_crop = transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 0.9))
color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

combined_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

# --- 3. 執行轉換並視覺化 ---
print(f"Loading original image from: {IMAGE_PATH}")
original_img = Image.open(IMAGE_PATH).convert('RGB')
resized_original = transforms.Resize((IMG_SIZE, IMG_SIZE))(original_img)

flipped_img = horizontal_flip(resized_original)
cropped_img = random_crop(original_img)
jittered_img = color_jitter(resized_original)
combined_img1 = combined_transform(original_img)
combined_img2 = combined_transform(original_img)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Data Augmentation Examples', fontsize=24)

images = [resized_original, flipped_img, cropped_img, jittered_img, combined_img1, combined_img2]
# *** 核心修改點：更新標題為您指定的英文描述 ***
titles = [
    'Original Image (Resized)',
    'Horizontal Flip',
    'Random Resized Crop',
    'Color Jitter',
    'Combination of Crop, Flip, Jitter #1',
    'Combination of Crop, Flip, Jitter #2'
]

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.set_title(titles[i], fontsize=16)
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(SAVE_PATH)

print(f"\nSuccessfully generated and saved detailed augmentation visualization to: {SAVE_PATH}")