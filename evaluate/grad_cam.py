# Copyright (c) 2025 YI-AN YEH
# This project is licensed under the MIT License - see the LICENSE file for details.

# C:\Code\ClassHomeWork\plantdisease\evaluate\grad_cam.py

"""
Grad-CAM visualization script for our champion model: Fine-tuned ResNet50.

This script loads the best fine-tuned ResNet50 model and a few sample images,
then generates Grad-CAM heatmaps to visualize model focus.

The output filename will include the original image's filename for easy traceability.
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2 # OpenCV for image manipulation

# --- 1. 路徑與核心定義 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 為了重複使用程式碼，我們從 evaluate_All.py 導入需要的類別和函數
# 這要求 evaluate_All.py 和 grad_cam.py 在同一個資料夾內
try:
    from evaluate_All import get_model, PlantDiseaseDataset
except ImportError:
    print("Error: Could not import from evaluate_All.py. Make sure both scripts are in the same directory.")
    sys.exit(1)

IMG_SIZE = 224
NUM_CLASSES = 15

# --- 2. Grad-CAM 核心實現 ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        # 為了能處理 .eval() 模式下的 backward pass，我們使用 register_full_backward_hook
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, index=None):
        self.model.eval()
        output = self.model(x)
        
        if index is None:
            index = torch.argmax(output, dim=1).item()
            
        one_hot = torch.zeros_like(output)
        one_hot[0][index] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8) # 加上一個極小值避免除以零
        
        return cam.squeeze().cpu().detach().numpy()

# --- 3. 視覺化輔助函數 ---
def show_cam_on_image(img, mask, save_path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = np.float32(heatmap) * 0.4 + np.float32(img) * 0.6
    superimposed_img = np.uint8(superimposed_img)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(superimposed_img)
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Grad-CAM image saved to: {save_path}")

# --- 4. 主執行函數 ---
def main(args):
    # --- 載入冠軍模型 ---
    model_name = 'resnet50_finetune'
    fold = args.fold
    
    config = {
        'dir': 'train_ResNet50_FineTune_output',
        'prefix': 'resnet50_finetune_model_fold_'
    }
    
    model_dir = os.path.join(PROJECT_ROOT, 'output', config['dir'])
    model_path = os.path.join(model_dir, f"{config['prefix']}{fold}_best.pth")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading champion model: {model_path}")
    
    model = get_model(model_name, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # --- 選擇 Grad-CAM 的目標層 ---
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    # --- 載入資料 ---
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'test_set.csv'))
    json_path = os.path.join(PROJECT_ROOT, 'label_mapping.json')
    with open(json_path, 'r') as f:
        label_mapping = {int(k): v for k, v in json.load(f).items()}

    # --- 準備圖像預處理 ---
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # --- 隨機選取 N 張圖片進行視覺化 ---
    print(f"\nGenerating {args.num_images} random Grad-CAM visualizations...")
    for i in range(args.num_images):
        random_idx = random.randint(0, len(test_df) - 1)
        sample = test_df.iloc[random_idx]
        img_path = os.path.join(PROJECT_ROOT, sample['filepath'])
        true_label_idx = sample['label_idx']
        
        # *** 新增：獲取原始檔名 ***
        original_filename_with_ext = os.path.basename(sample['filepath'])
        original_filename, _ = os.path.splitext(original_filename_with_ext)

        original_img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        input_tensor = preprocess(original_img).unsqueeze(0).to(device)

        mask = grad_cam(input_tensor)
        
        output = model(input_tensor)
        pred_label_idx = torch.argmax(output, dim=1).item()
        
        result_text = "Correct" if pred_label_idx == true_label_idx else "WRONG"
        
        save_filename = f"gradcam_{original_filename}_{result_text}.png"
        save_path = os.path.join(model_dir, save_filename)
        
        show_cam_on_image(np.array(original_img), mask, save_path)
        
# --- 5. 程式進入點 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for the champion model.")
    parser.add_argument('--fold', type=int, required=True, 
                        help='The best fold number of the champion model (resnet50_finetune).')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of random images to visualize.')
    
    args = parser.parse_args()
    main(args)